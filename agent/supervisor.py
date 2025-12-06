"""
Multi-Agent Supervisor for Healthcare Data Retrieval.

Orchestrates specialized agents to answer medical queries by coordinating between:
- EHR Agent: Structured data (SQL queries for demographics, labs, diagnoses)
- Reports Agent: Unstructured data (clinical notes, radiology reports)
- Synthesizer: Combines both sources into coherent answers

Key Features:
- Context-aware routing (distinguishes patient/admission/report IDs)
- Multi-step workflow handling (automatically sequences agent calls)
- Loop prevention (avoids redundant data fetching)
- RAGAS evaluation support (tracks retrieval contexts)
"""

import os
import re
import pandas as pd
from typing import TypedDict, List, Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from agent.ehr_agent import ehr_graph
from agent.reports_agent import reports_graph
from src.utils import mask_pii


load_dotenv()

# Supervisor State
class SupervisorState(TypedDict):
    """
    State schema for supervisor workflow.
    
    Tracks conversation history, routing decisions, cached data,
    and patient context across multi-turn interactions.
    """
    messages: List[BaseMessage]
    next_step: str
    worker_input: str
    
    # Context Data
    sql_data: Optional[str]
    report_data: Optional[str]
    target_subject_ids: Optional[List[int]]
    ragas_context: List[str]
    
    dialogue_status: Literal["complete", "in_progress"]
    last_processed_query: Optional[str]

# Models and Helpers
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
classifier_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class TaskAllocation(BaseModel):
    """Decides the next worker based on current data gaps."""
    reasoning: str = Field(description="Analyze gaps between user request and current data.")
    next_worker: Literal["EHR", "REPORTS", "FINISH"]
    specific_instruction: str = Field(description="The specific query for the worker.")

class ContextDecision(BaseModel):
    """
    Analyzes the user's query to classify numbers and intent based on surrounding text.
    """
    is_new_patient: bool = Field(
        description="True ONLY if the user is switching to a different PATIENT context."
    )
    detected_id: Optional[int] = Field(
        description="The specific number extracted from the text."
    )
    id_type: Literal["patient_id", "admission_id", "report_id", "ambiguous"] = Field(
        description="Classify what the detected_id represents based on the sentence structure."
    )
    reasoning: str

def format_context_for_ragas(df: pd.DataFrame) -> str:
    """
    Transforms raw SQL dataframe into 'LLM-readable' natural language strings 
    to prevent Faithfulness errors in Ragas evaluation.
    """
    if df.empty:
        return "No results found."

    df_copy = df.copy()
    
    if 'gender' in df_copy.columns:
        df_copy['gender'] = df_copy['gender'].replace({'F': 'Female', 'M': 'Male'})
        
    if 'admission_type' in df_copy.columns:
        df_copy['admission_type'] = df_copy['admission_type'].replace({
            'EW EMER.': 'Emergency (EW EMER.)',
            'URGENT': 'Urgent',
            'ELECTIVE': 'Elective',
            'SURGICAL SAME DAY ADMISSION': 'Surgical Same Day'
        })

    for col in df_copy.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                df_copy[col] = df_copy[col].dt.strftime('%B %d, %Y at %H:%M')
            except Exception:
                pass

    return df_copy.to_string(index=False)

def supervisor_node(state: SupervisorState):
    """
    Orchestrates workflow by analyzing data gaps and routing to workers.
    
    Determines if query requires EHR data, reports, or both, while
    maintaining patient context and preventing redundant agent calls.
    """
    print("\n[Supervisor] Planning next step...")
    
    messages = state['messages']
    current_status = state.get("dialogue_status", "complete")
    
    sql_context = state.get('sql_data', "") or ""
    report_context = state.get('report_data', "") or ""
    current_ids = state.get('target_subject_ids', []) or []
    current_ragas_context = state.get('ragas_context', []) or []
    
    current_user_query = messages[-1].content if messages else ""

    if current_status == "complete":
        print(f"   [Supervisor] New Turn: '{current_user_query}'")
        
        should_wipe = False
        
        candidate_numbers = re.findall(r'\b\d{7,8}\b', current_user_query)
        if candidate_numbers or current_ids:
            check_prompt = f"""
Previous Query: "{state.get('last_processed_query', '')}"
Current Query: "{current_user_query}"
Current Active Patient IDs: {current_ids}
Candidate Numbers Found: {candidate_numbers}

YOUR TASK:
1. Analyze the context surrounding the numbers in 'Current Query'.
2. Determine if the user is switching to a NEW PATIENT or asking about a specific ADMISSION/REPORT.

RULES FOR CLASSIFICATION (Based on Text Context ONLY):
- **PATIENT ID (`subject_id`)**: 
  - Look for keywords: "Patient", "Subject", "Person", "Him", "Her".
  - If the user provides a number with NO context (e.g. "Search 10002"), assume it is a PATIENT ID.

- **ADMISSION ID (`hadm_id`)**: 
  - Look for keywords: "Admission", "Visit", "Stay", "Discharge".
  - Example: "Check admission 225..." -> This is an admission, NOT a new patient.

- **REPORT ID (`note_id`)**:
  - Look for keywords: "Report", "Note", "Scan", "Radiology", "X-Ray".
  - Example: "Read report 5555..." -> This is a document, NOT a new patient.

DECISION LOGIC:
- If `id_type` is 'patient_id' AND it is different from Active ID -> `is_new_patient` = True.
- If `id_type` is 'admission_id' OR 'report_id' -> `is_new_patient` = False (Stay on current patient).
- If NO numbers found but text implies follow-up (e.g. "his age") -> `is_new_patient` = False.
"""
            
            decision = classifier_llm.with_structured_output(ContextDecision).invoke(check_prompt)
            print(f"   [Supervisor] Analysis: {decision.id_type.upper()} ({decision.detected_id}) -> {decision.reasoning}")

            if decision.is_new_patient:
                print(f"   [Supervisor] Switching to New Patient: {decision.detected_id}")
                should_wipe = True
                if decision.detected_id:
                    current_ids = [decision.detected_id]
                else:
                    current_ids = []

            elif decision.id_type in ["admission_id", "report_id"]:
                print(f"   [Supervisor] Focusing on {decision.id_type} {decision.detected_id}. Keeping Patient Context.")
                should_wipe = False
                
            elif decision.detected_id and decision.id_type == "patient_id" and decision.detected_id not in current_ids:
                print(f"   [Supervisor] Patient ID Mismatch. Switching to {decision.detected_id}.")
                should_wipe = True
                current_ids = [decision.detected_id]

            elif decision.id_type == "ambiguous" and not decision.detected_id:
                print("   [Supervisor] New search query detected. Wiping context.")
                should_wipe = True
                current_ids = []
                
            else:
                print("   [Supervisor] Context Preserved (Follow-Up).")
                should_wipe = False

        if should_wipe:
            print("   [Supervisor] Wiping Context.")
            sql_context = ""
            report_context = ""
            current_ragas_context = []


    has_sql_data = len(sql_context) > 5
    has_report_data = len(report_context) > 5
    has_patient_ids = len(current_ids) > 0

    system_prompt = f"""
You are the Hospital Administrator coordinating data retrieval.

CURRENT DATA STATUS:
- Structured Data (SQL): {"AVAILABLE" if has_sql_data else "MISSING"}
- Text Data (Reports): {"AVAILABLE" if has_report_data else "MISSING"}
- Patient IDs Identified: {current_ids if has_patient_ids else "None"}

USER REQUEST: {current_user_query}

DECISION RULES FOR MULTI-STEP QUERIES:

1. **If request needs BOTH structured AND unstructured data:**
   - FIRST: Call 'EHR' to identify patients and get structured data
   - THEN: Call 'REPORTS' to get clinical narratives/radiology
   - Example: "Show me patient demographics and their imaging results"
     → Step 1: EHR (demographics)
     → Step 2: REPORTS (imaging)

2. **If you already have SQL data but request asks for reports/radiology/notes:**
   → Call 'REPORTS' (DO NOT call EHR again)
   - Keywords indicating REPORTS needed: "radiology", "report", "scan", "imaging", "physician notes", "discharge summary", "clinical text"

3. **If you already have report data but need demographics/labs/vitals:**
   → Call 'EHR'
   - Keywords indicating EHR needed: "age", "admission date", "lab results", "vitals", "demographics", "diagnosis codes"

4. **If you have ALL data needed to answer the question:**
   → Call 'FINISH'

CRITICAL PATTERNS TO RECOGNIZE:
- Query mentions BOTH structured info AND clinical text → TWO-STEP workflow
- "What were the vital signs and what did the X-ray show?" → EHR first, then REPORTS
- "Show me the CT scan for patient 12345" → If you have patient ID, go to REPORTS
- "Patient's lab values and discharge notes" → EHR first, then REPORTS

CURRENT SITUATION ANALYSIS:
- Already retrieved SQL data: {has_sql_data}
- Already retrieved Reports: {has_report_data}
- Patient IDs available: {has_patient_ids}

TASK: Based on the current data status and user request, what is the NEXT SINGLE STEP?
Think step-by-step:
1. What data do I already have?
2. What data is still missing to answer the question?
3. Which agent should I call next to fill that gap?
"""
    
    input_msgs = [SystemMessage(content=system_prompt)] + messages[-1:]
    decision = llm.with_structured_output(TaskAllocation).invoke(input_msgs)
    
    next_worker = decision.next_worker
    specific_instruction = decision.specific_instruction

    if next_worker == "EHR" and has_sql_data:
        print("   [Supervisor] Loop Prevented: Already have SQL data.")
        
        reports_keywords = ["report", "radiology", "scan", "imaging", "notes", "physician", "discharge summary", "clinical text"]
        needs_reports = any(keyword in current_user_query.lower() for keyword in reports_keywords)
        
        if needs_reports and not has_report_data:
            print("   [Supervisor] Query needs reports → Switching to REPORTS agent.")
            next_worker = "REPORTS"
            specific_instruction = f"Find relevant clinical reports/radiology for patient(s): {current_ids}"
        else:
            next_worker = "FINISH"
    
    if next_worker == "REPORTS" and has_report_data:
        print("   [Supervisor] Loop Prevented: Already have Reports.")
        next_worker = "FINISH"
        
    print(f"   -> Decision: {next_worker}")
    print(f"   -> Instruction: {specific_instruction}")
    
    update = {
        "next_step": next_worker,
        "worker_input": specific_instruction,
        "dialogue_status": "in_progress",
        "last_processed_query": current_user_query,
        "sql_data": sql_context,
        "report_data": report_context,
        "target_subject_ids": current_ids,
        "ragas_context": current_ragas_context
    }
    
    return update


def call_ehr_agent(state: SupervisorState):
    """
    Fetches structured EHR data via SQL queries.
    
    Returns demographics, diagnoses, labs, and vitals. Extracts patient IDs
    from results and formats data for RAGAS evaluation.
    """
    print("\n[Supervisor] Calling EHR Agent...")
    
    if ehr_graph is None:
        raise RuntimeError("EHR agent failed to import. Check import errors above.")
    
    inputs = {
        "question": state['worker_input'], 
        "filter_subject_ids": state.get('target_subject_ids'),
        "retries": 0, "investigated": False
    }
    result = ehr_graph.invoke(inputs)
    
    raw_data = result.get('query_result', [])
    df = pd.DataFrame(raw_data) if raw_data else pd.DataFrame()
    
    output_str = "No results found."
    ragas_output_str = "No results found." 
    extracted_ids = []
    
    current_ids = state.get('target_subject_ids', [])
    
    if not df.empty:
        output_str = df.to_string(index=False, header=True)

        humanized_table = format_context_for_ragas(df)

        if current_ids:
            ragas_output_str = f"SQL Query Results for Patient IDs {current_ids}:\n{humanized_table}"
        else:
            ragas_output_str = f"SQL Query Results:\n{humanized_table}"

        if 'subject_id' in df.columns:
            try:
                raw_ids = df['subject_id'].dropna().unique().tolist()
                extracted_ids = [int(x) for x in raw_ids if str(x).isdigit()]
            except: 
                pass
    else:
        if current_ids:
            ragas_output_str = f"""SQL Query Results for Patient IDs {current_ids}:
No matching records found in the database.

INTERPRETATION: This is a definitive NEGATIVE finding. The queried diagnosis/medication/condition is NOT present in the patient's structured medical records.
"""
        else:
            ragas_output_str = """SQL Query Results:
No matching records found in the database.

INTERPRETATION: This is a definitive NEGATIVE finding. The queried data does not exist in the structured medical records.
"""
            
    existing_ids = state.get('target_subject_ids') or []
    final_ids = list(set(existing_ids + extracted_ids))
    
    current_ragas = state.get('ragas_context', [])
    new_ragas = current_ragas + [ragas_output_str]
    
    return {
        "sql_data": output_str,
        "target_subject_ids": final_ids,
        "ragas_context": new_ragas
    }


def call_reports_agent(state: SupervisorState):
    """
    Retrieves unstructured clinical documents via vector search.
    
    Returns radiology reports, physician notes, and discharge summaries.
    Formats documents with metadata for RAGAS evaluation.
    """
    print("\n[Supervisor] Calling Reports Agent...")
    
    inputs = {
        "question": state['worker_input'], 
        "filter_subject_ids": state.get('target_subject_ids'),
        "chat_history": [m.content for m in state['messages']]
    }
    
    result = reports_graph.invoke(inputs)
    
    raw_docs = result.get('documents', [])
    ragas_docs_list = []
    
    for doc in raw_docs:
        if hasattr(doc, 'page_content'):
            source = doc.metadata.get('source', 'Unknown Report')
            content = f"Source: {source}\nContent: {doc.page_content}"
            ragas_docs_list.append(content)
        else:
            ragas_docs_list.append(str(doc))
    
    if ragas_docs_list:
        current_ids = state.get('target_subject_ids', [])
        worker_query = state.get('worker_input', '')
        
        if current_ids:
            context_header = f"Clinical Reports for Patient {current_ids[0]}:\n"
        else:
            context_header = "Clinical Reports Search Results:\n"
        
        ragas_docs_list[0] = f"{context_header}Query: {worker_query}\n\n{ragas_docs_list[0]}"
    
    current_ragas = state.get('ragas_context', [])
    new_ragas = current_ragas + ragas_docs_list
    
    return {
        "report_data": result.get('candidate_answer', "No info found."),
        "target_subject_ids": state.get('target_subject_ids'),
        "ragas_context": new_ragas
    }

def synthesizer_node(state: SupervisorState):
    """
    Combines SQL and report data into comprehensive clinical answer.
    
    Handles negative findings, validates cross-source consistency,
    and provides evidence-based responses with proper citations.
    """
    print("\n[Supervisor] Synthesizing Final Answer...")
    
    sql_context = state.get('sql_data', "N/A")
    report_context = state.get('report_data', "N/A")
    final_ragas_context = state.get('ragas_context', [])

    user_q = [m for m in state['messages'] if isinstance(m, HumanMessage)][-1].content
    
    prompt = f"""
You are a clinical data synthesizer combining structured (SQL) and unstructured (Radiology) data.

USER QUESTION: {user_q}

RETRIEVED DATA:

STRUCTURED SQL DATA:{sql_context}

CLINICAL REPORTS DATA:{report_context}

CRITICAL INSTRUCTIONS FOR HIGH-QUALITY ANSWERS:

1. ANSWER STRUCTURE:
   - Start with direct answer (Yes/No for yes/no questions)
   - Support with specific evidence from BOTH SQL and Reports
   - End concisely - no unnecessary elaboration

2. HANDLING NEGATIVE FINDINGS (CRITICAL FOR RAGAS CONTEXT RECALL):
   SQL shows "No results found" → State: "No, the patient does NOT have [X]. This is confirmed by the absence of records in the diagnoses table."
   Report says "no evidence of X", "no X seen", "X is unremarkable", "X is normal" → State: "No [X] found. The radiology report explicitly states '[exact quote]' (Note ID: XXX)"
   NEVER say "information not found" if you have explicit negative evidence
   Only say "information not found" if BOTH SQL and reports contain no relevant data at all

3. VALIDATION QUERIES (when asked to "confirm" or "check"):
   - Compare SQL data with Report data explicitly
   - If SQL says "No diagnosis code" but report mentions findings → Explain: "The diagnosis is not formally coded, but radiology shows..."
   - If both agree → State: "Confirmed. Both structured records and clinical reports indicate..."

4. ICD CODE QUERIES:
   - Quote exact ICD codes from SQL (e.g., "ICD 07054")
   - Match with clinical description from long_title field
   - If code exists but reports don't mention → State both facts

5. RADIOLOGY FINDING QUERIES:
   - ALWAYS cite with (Note ID: XXX-RR-YY)
   - Quote exact medical phrases from reports
   - Include measurements when available (e.g., "3.6 cm nodule")

6. TIME/DURATION CONVERSIONS:
   - Decimal days → hours/minutes (e.g., 0.5 days = 12 hours)
   - Include both formats when relevant

7. MEDICAL TERMINOLOGY:
   - Use EXACT terms from reports (don't paraphrase "ascites" as "fluid")
   - Preserve clinical abbreviations (e.g., "c/b" = "complicated by")

8. COMPARATIVE QUESTIONS:
   - SQL vs Reports: Explicitly state what each source shows
   - Example: "The diagnosis table shows ICD 5723 (Portal Hypertension). This is supported by radiology findings of ascites and splenomegaly (Note ID: 10000032-RR-15)."

EXAMPLES OF GOOD ANSWERS:

Example 1 (Negative Finding):
Q: "Does patient have Pneumonia?"
SQL: No results found
Report: "no focal consolidation"
GOOD: "No. The patient does NOT have Pneumonia. This is confirmed by: (1) SQL: No ICD codes for Pneumonia in the diagnoses table, (2) Radiology: Report 0 explicitly states 'no focal consolidation' (Note ID: 10000032-RR-14), which rules out acute pneumonia."

Example 2 (Positive Finding with Validation):
Q: "Confirm primary diagnosis"
SQL: ICD 07054 (Chronic Hepatitis C)
Report: Mentions "cirrhosis, ascites, splenomegaly"
GOOD: "The primary diagnosis is Chronic Hepatitis C (ICD 07054). This is validated by radiology reports showing classic complications: cirrhosis, ascites, and splenomegaly (Note ID: 10000032-RR-15, 10000032-RR-47)."

Example 3 (Ambiguity Resolution):
Q: "Is the free fluid ascites or hemorrhage?"
SQL: ICD 7895 (Ascites)
Report: "mild abdominal and pelvic ascites"
GOOD: "The free fluid is ascites (cirrhosis-related). This is confirmed by: (1) SQL: ICD Code 7895 (Ascites) is present, (2) Radiology: Report explicitly describes 'mild abdominal and pelvic ascites' (Note ID: 10039708-RR-85), with no mention of hemorrhage or bleeding."

NOW ANSWER THE USER'S QUESTION:
"""
    
    final_response = llm.invoke(prompt)
    
    return {
        "messages": [final_response],
        "sql_data": "", 
        "report_data": "",
        "dialogue_status": "complete",
        "ragas_context": final_ragas_context
    }


# Build workflow graph
workflow = StateGraph(SupervisorState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("ehr_worker", call_ehr_agent)
workflow.add_node("reports_worker", call_reports_agent)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("supervisor")

def route_next(state: SupervisorState):
    return state['next_step']

workflow.add_conditional_edges("supervisor", route_next, {
    "EHR": "ehr_worker", 
    "REPORTS": "reports_worker", 
    "FINISH": "synthesizer"
})
workflow.add_edge("ehr_worker", "supervisor")
workflow.add_edge("reports_worker", "supervisor")
workflow.add_edge("synthesizer", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    
    config = {"configurable": {"thread_id": "cli_session_v1"}}
    ragas_dataset_rows = []

    while True:
        try:
            q = input("\nAdmin: How can I help? (or 'q'): ").strip()
            if q.lower() in ['q', 'quit']: break
            if not q: continue
            
            result = app.invoke({"messages": [HumanMessage(content=q)]}, config=config)
            
            agent_answer = result['messages'][-1].content
            retrieved_contexts = result.get('ragas_context', [])

            if retrieved_contexts:
                ragas_row = {
                    "question": q,
                    "answer": agent_answer,
                    "contexts": retrieved_contexts,
                    "ground_truth": "PENDING_HUMAN_REVIEW" 
                }
                ragas_dataset_rows.append(ragas_row)
                print(f"\n[Ragas] Captured {len(retrieved_contexts)} context chunks for evaluation.")

            print("\n" + "="*50)
            print("FINAL RESPONSE:")
            print(agent_answer)
            print("="*50)
            
        except Exception as e:
            print(f"Error: {e}")