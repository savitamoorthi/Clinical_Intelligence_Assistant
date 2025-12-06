
"""
Radiology Reports Agent
A LangGraph-based agent for retrieving and analyzing radiology reports from a vector database.

Features:
- Context-aware query reformulation
- Intelligent filter extraction for patient/admission/note IDs
- Hybrid retrieval: vector search + cross-encoder reranking
- Relevance grading and citation validation
- Full document retrieval for specific report queries
"""

import os
from typing import TypedDict, List, Optional, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.messages import SystemMessage

load_dotenv()

# Database Configuration
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "radiology_knowledge_base"

# Model Configuration
MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'

print("   [Reports] Connecting to Zilliz...")
connections.connect("default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
collection = Collection(COLLECTION_NAME)
collection.load()

print("   [Reports] Loading Embedding Models...")
embedding_model = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


class ReportsAgentState(TypedDict):
    """
    Shared workflow state for the Radiology Reports Agent.

    Attributes:
        question: Current user question (possibly contextualized)
        chat_history: Log of prior conversational turns
        filter_subject_ids: IDs optionally injected by a supervising agent
        generated_filters: Extracted patient/admission/note/date filters
        documents: Retrieved text chunks or full documents
        candidate_answer: Generated LLM answer
        retries: Retry counter for missing citations
    """
    question: str
    chat_history: List[str]
    filter_subject_ids: Optional[List[int]] 
    generated_filters: dict
    documents: List[str]
    candidate_answer: str
    retries: int


def contextualize_node(state: ReportsAgentState):
    """
    Convert follow-up questions into standalone queries when necessary.
    - Preserves all identifiers.
    - Skips contextualization if explicit IDs already appear.
    """
    print("   [Reports] 0. Contextualize...")
    question = state['question']
    history = state.get('chat_history', [])

    has_ids = any(keyword in question.lower() for keyword in ['note id', 'patient', 'report', 'admission'])
    
    if not history or has_ids:
        print(f"      Keeping original: '{question}'")
        return {"question": question}
    
    history_str = "\n".join(history[-6:])
    system_prompt = f"""
Rewrite the user's question to be STANDALONE based on history.
IMPORTANT: Preserve ALL specific identifiers (patient IDs, note IDs, dates, etc.)

History: {history_str}
Follow-up: {question}

Rewritten (standalone):
"""
    rewritten = llm.invoke(system_prompt).content
    print(f"      Rewritten: {rewritten}")
    return {"question": rewritten}


class SearchFilters(BaseModel):
    """
    Schema for extracting structured identifiers from natural language queries.
    """
    subject_id: Optional[int] = Field(None, description="Patient ID")
    hadm_id: Optional[int] = Field(None, description="Admission ID")
    note_id: Optional[str] = Field(None, description="Specific Report/Note ID (e.g. 10000032-RR-14)")
    target_date: Optional[str] = Field(None, description="Specific date mentioned (YYYY-MM-DD format)")


def router_node(state: ReportsAgentState):
    """
    Extract patient IDs, admission IDs, note IDs, and dates using the LLM schema.
    - Merges filters from supervising agent if provided.
    """
    print("   [Reports] 1. Router...")
    
    passed_ids = state.get('filter_subject_ids')
    structured_llm = llm.with_structured_output(SearchFilters)
    
    extraction_prompt = f"""
Extract medical identifiers from this query. Be very careful to catch Note IDs!

Query: "{state['question']}"

EXTRACTION RULES:
- "Note ID XXX-RR-YY" or "report XXX-RR-YY" or "scan XXX-RR-YY" → extract as note_id
- "Report 0", "Report 19" → map to note_id pattern for this patient
- "Patient XXX" or "subject XXX" → extract as subject_id
- "Admission XXX" → extract as hadm_id
- Dates → extract as target_date (YYYY-MM-DD)

CRITICAL: If you see a pattern like "10000032-RR-15" anywhere in the text, extract it as note_id!

If multiple IDs are present, extract ALL of them.
"""
    
    result = structured_llm.invoke(extraction_prompt)
    
    if isinstance(result, dict):
        data = result
    else:
        data = result.model_dump()
        
    filters = {k: v for k, v in data.items() if v is not None}
    

    if passed_ids and len(passed_ids) > 0:
        filters['subject_id_list'] = passed_ids
        if 'subject_id' not in filters and len(passed_ids) == 1:
            filters['subject_id'] = passed_ids[0]
        print(f"      Using IDs from Supervisor: {passed_ids}")
    
    print(f"      Final Filters: {filters}")
    return {"generated_filters": filters}


def retriever_node(state: ReportsAgentState):
    """
    Perform hybrid retrieval:
    1. Vector search using embeddings
    2. Cross-encoder reranking
    3. Adaptive top-K selection
    4. Full-document expansion when specific note IDs are requested
    """
    print("   [Reports] 2. Retriever...")
    question = state['question']
    filters = state['generated_filters']
    
    print(f"      Query: '{question}'")
    print(f"      Extracted Filters: {filters}")
    
    expr_parts = []
    

    if 'subject_id_list' in filters:
        ids = filters['subject_id_list']
        if len(ids) > 0:
            expr_parts.append(f"subject_id in {ids}")
    elif filters.get('subject_id'):
        expr_parts.append(f"subject_id == {filters['subject_id']}")


    if filters.get('hadm_id'):
        expr_parts.append(f"hadm_id == {filters['hadm_id']}")


    specific_note_requested = False
    if filters.get('note_id'):
        val = filters['note_id']
        specific_note_requested = True
        if len(val) < 5:
            expr_parts.append(f"note_id like '%-{val}'")
        else:
            expr_parts.append(f'(note_id == "{val}" or note_id like "%{val}")')
    

    if filters.get('target_date'):
        date_val = filters['target_date']
        expr_parts.append(f"charttime like '{date_val}%'")

    filter_expr = " and ".join(expr_parts) if expr_parts else ""
    print(f"      Filter Expression: {filter_expr if filter_expr else 'None'}")


    query_vector = embedding_model.encode([question])[0].tolist()
    
    initial_results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {}},
        limit=30, 
        expr=filter_expr,
        output_fields=["note_id", "text_chunk", "charttime"]
    )
    
    if not initial_results or not initial_results[0]:
        print(f"No results returned from vector search")
        return {"documents": []}
    
    print(f"Retrieved {len(initial_results[0])} initial results")


    hits = initial_results[0]
    cross_input = [[question, hit.entity.get('text_chunk')] for hit in hits]
    scores = reranker.predict(cross_input)
    
    scored_hits = [{"score": scores[i], "note_id": hit.entity.get('note_id'), 
                    "date": hit.entity.get('charttime'),
                    "text_chunk": hit.entity.get('text_chunk')} 
                   for i, hit in enumerate(hits)]
    

    sorted_hits = sorted(scored_hits, key=lambda x: x['score'], reverse=True)
    

    specific_note_requested = bool(filters.get('note_id'))
    
    if specific_note_requested:
        top_k = 3  
        print(f"Specific note request detected - using top_k={top_k}")
    else:
        top_k = 10  
        print(f"General query - returning focused chunks (top_k={top_k})")
    
    top_hits = sorted_hits[:top_k]
    
    if not top_hits:
        print(f"No hits after selection")
        return {"documents": []}
    
    print(f"Selected top {len(top_hits)} results")
    print(f"Score range: {top_hits[0]['score']:.4f} to {top_hits[-1]['score']:.4f}")
    print(f"Top Note IDs: {list(set([h['note_id'] for h in top_hits]))}")

    if specific_note_requested and filters.get('note_id'):
        print(f"Specific note requested - fetching FULL document")
        
        relevant_note_ids = list(set([h['note_id'] for h in top_hits]))
        ids_str = ", ".join([f"'{nid}'" for nid in relevant_note_ids])
        expansion_expr = f"note_id in [{ids_str}]"
          
        full_chunks = collection.query(
            expr=expansion_expr,
            output_fields=["note_id", "text_chunk", "charttime"],
            limit=200
        )
        
        reports_map = {}
        for item in full_chunks:
            nid = item['note_id']
            if nid not in reports_map:
                reports_map[nid] = {"date": item['charttime'], "chunks": []}
            reports_map[nid]["chunks"].append(item['text_chunk'])
        
        formatted_docs = []
        for nid, data in reports_map.items():
            full_text = "\n".join(data["chunks"])
            formatted_docs.append(
                f"[Note ID: {nid} | Date: {data['date']}]\n"
                f"CONTENT:\n{full_text}\n"
            )
        
        print(f"Returning {len(formatted_docs)} FULL documents")
    
    else:
        print(f"General query - returning focused chunks")
        formatted_docs = []
        for hit in top_hits:
            formatted_docs.append(
                f"[Note ID: {hit['note_id']} | Date: {hit['date']}]\n"
                f"CONTENT:\n{hit['text_chunk']}\n"
            )
        print(f"      Returning {len(formatted_docs)} focused chunks")
    
    return {"documents": formatted_docs}


class Grade(BaseModel):
    """Structured output for the relevance grader."""
    is_relevant: bool = Field(description="Are documents relevant?")


def grader_node(state: ReportsAgentState):
    """
    Determine whether retrieved documents are relevant to the question.
    - Skips grading for ID-based retrieval.
    - Uses LLM to validate semantic relevance.
    """
    print("   [Reports] 3. Grader...")
    docs = state['documents']
    if not docs: return {"documents": []}
    
    filters = state.get('generated_filters', {})

    if (filters.get('note_id') or 
        filters.get('target_date') or 
        filters.get('subject_id') or 
        filters.get('subject_id_list') or 
        filters.get('hadm_id')):
        print("Skipping Grader (ID-based search - trusting retrieval).")
        return {}
    

    print("Grading semantic relevance...")
    context_snippet = "\n\n".join(docs)[:15000]
    
    grader_prompt = f"""
Question: {state['question']}

Retrieved Chunks: {context_snippet}

Does AT LEAST ONE chunk contain information that could help answer the question?
- If you see ANY relevant medical terms, patient info, or clinical findings → RELEVANT
- Only mark IRRELEVANT if completely off-topic (e.g., liver question getting orthopedic reports)

Be generous - partial matches count as relevant.
"""
    
    grader_llm = llm.with_structured_output(Grade)
    grade = grader_llm.invoke(grader_prompt)
    
    is_relevant = grade.is_relevant if hasattr(grade, 'is_relevant') else grade.get('is_relevant', False)
    
    if not is_relevant:
        print("Documents marked IRRELEVANT. Clearing.")
        return {"documents": []}
    
    print("Documents marked RELEVANT.")
    return {}


def generator_node(state: ReportsAgentState):
    """
    Generate a clinical answer using ONLY the retrieved radiology text.
    - Enforces strict negative-finding interpretation rules.
    - Ensures terminology fidelity and citation format.
    """
    print("   [Reports] 4. Generator...")
    docs = state['documents']
    
    if not docs:
        return {
            "candidate_answer": "No relevant radiology reports found.",
            "retries": state.get("retries", 0)
        }
    
    context_str = "\n\n".join(docs)
    
    prompt = f"""
You are a clinical radiology assistant. Answer ONLY using the provided context.

CONTEXT:
{context_str}

QUESTION: {state['question']}

CRITICAL INSTRUCTIONS FOR NEGATIVE FINDINGS:
1. If you see "no evidence of X", "no X identified", "X is unremarkable", "X is normal" → This means X is ABSENT
2. This is a VALID ANSWER - state clearly: "No [X] found" or "[X] is absent"
3. NEVER say "information not found" if you have explicit negative evidence
4. Examples:
   - Report says "no focal consolidation" → Answer: "No focal consolidation (rules out pneumonia)"
   - Report says "heart size is normal" → Answer: "Heart size is normal (no cardiomegaly)"

POSITIVE FINDINGS:
5. ALWAYS cite sources using (Note ID: XXX-RR-YY) format for ANY clinical finding
6. Use EXACT medical terminology from reports - don't paraphrase
7. Include specific measurements when available (e.g., "3.6 cm nodule")

ANSWER STRUCTURE:
8. For yes/no questions: Start with "Yes" or "No" followed by supporting evidence
9. For "what" questions: Provide direct answer with citations
10. For "does report mention" questions: Quote the exact phrase with Note ID

ONLY if context has NO relevant information at all → state: "Information not found in available reports"

ANSWER:
"""
    response = llm.invoke(prompt)
    return {"candidate_answer": response.content}


def validator_node(state: ReportsAgentState):
    """
    Validate that answers comply with clinical citation rules.
    - Negative findings do not require citation.
    - Positive findings MUST include "(Note ID: ...)".
    - Triggers retry if citation is missing.
    """
    print("   [Reports] 5. Validator...")
    answer = state['candidate_answer']
    
    negative_phrases = [
        "no relevant", "cannot find", "not found", "information not found",
        "no mention", "no evidence", "no reports", "absent", "unremarkable",
        "normal", "no focal", "no acute", "no [", "does not"
    ]
    
    if any(phrase in answer.lower() for phrase in negative_phrases):
        print("      Negative result - citation not required.")
        return {}
    
    # For positive findings, require citation
    if "(Note ID:" not in answer:
        print("      CITATION MISSING.")
        return {
            "candidate_answer": "ERROR_CITATION_MISSING",
            "retries": state.get("retries", 0) + 1
        }
    
    print("      Citation present.")
    return {}


# Graph Construction
def should_continue(state: ReportsAgentState):
    """
    Determines whether the workflow must retry (citation missing) or terminate.
    """
    if state['candidate_answer'] == "ERROR_CITATION_MISSING":
        if state['retries'] >= 3: return "end"
        return "retry"
    return "end"

workflow = StateGraph(ReportsAgentState)

workflow.add_node("contextualize", contextualize_node)
workflow.add_node("router", router_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("grader", grader_node)
workflow.add_node("generator", generator_node)
workflow.add_node("validator", validator_node)

workflow.set_entry_point("contextualize")
workflow.add_edge("contextualize", "router")
workflow.add_edge("router", "retriever")
workflow.add_edge("retriever", "grader")
workflow.add_edge("grader", "generator")
workflow.add_edge("generator", "validator")

workflow.add_conditional_edges(
    "validator",
    should_continue,
    {
        "retry": "generator",
        "end": END
    }
)

reports_graph = workflow.compile()