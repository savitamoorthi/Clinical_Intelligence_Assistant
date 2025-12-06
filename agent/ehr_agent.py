"""
EHR Agent: Text-to-SQL System for Medical Database Queries.

Converts natural language questions into SQL queries against MIMIC-III database.
Uses hybrid retrieval (schema graph + semantic search) to generate contextually
accurate queries with automatic error correction and empty result investigation.

Key Features:
- Schema-aware SQL generation with medical concepts
- Automatic error correction (syntax, ambiguous columns, table mismatches)
- Empty result investigation (fuzzy matching, alternative terms)
- Patient ID filtering for context-specific queries
- ICD code + clinical term dual-matching for comprehensive results

Workflow:
1. Retrieve: Find relevant tables/columns via hybrid search
2. Generate: Create SQL with enhanced medical query rules
3. Execute: Run query against database
4. Correct/Investigate: Fix errors or explore empty results
"""

import operator
from typing import Annotated, List, TypedDict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.retrieval import HybridRetriever
from src.db_graph import build_schema_graph
from src.utils import execute_query, sanitize_sql, mask_pii, lookup_values_in_db
from src.generation import generate_sql 
import yaml
import os
import re

load_dotenv()

# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    """
    State schema for EHR SQL generation workflow.
    
    Tracks question context, retrieval results, generated SQL,
    execution results, and error handling state.
    """
    question: str
    filter_subject_ids: Optional[List[int]]
    
    schema_context: str
    metrics_context: List[str]
    concepts_context: List[str]
    
    sql_query: str
    
    query_result: Optional[List[dict]] 
    
    error: str
    retries: int
    investigated: bool 

# Global Resources
print("   [EHR] Loading Resources...")
DB_PATH = 'data/mimic.db'
G = build_schema_graph(DB_PATH)
retriever = HybridRetriever(G)
llm_tool = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
print("   [EHR] Resources Loaded.")

def retrieve_node(state: AgentState):
    """
    Retrieves relevant schema context using hybrid search.
    
    Combines graph-based table relationships with semantic search
    to identify tables, columns, metrics, and medical concepts.
    """
    print(f"   [EHR] Searching for: {state['question']}")
    sub_graph, metrics, concepts = retriever.retrieve(state['question'])
    
    schema_text = []
    for node in sub_graph.nodes():
        if '.' not in node:
            cols = [n for n in sub_graph.neighbors(node) if '.' in n]
            schema_text.append(f"Table {node}: {', '.join(cols)}")
            
    return {
        "schema_context": "\n".join(schema_text),
        "metrics_context": metrics,
        "concepts_context": concepts,
        "retries": 0,
        "error": None,
        "investigated": False
    }

def generate_node(state: AgentState):
    """
    Generates SQL query with medical domain enhancements.
    
    Applies patient ID filters, ICD code fuzzy matching, and
    comprehensive search rules to maximize result recall.
    """
    print("   [EHR] Drafting SQL...")
    
    effective_question = state['question']
    passed_ids = state.get('filter_subject_ids')
    
    if passed_ids and len(passed_ids) > 0:
        print(f"   [EHR] Applied Supervisor Filter: Restricting to {len(passed_ids)} IDs.")
        ids_str = ", ".join(map(str, passed_ids))
        effective_question += f"\n\nCRITICAL: You MUST strictly filter results to ONLY include these subject_ids: ({ids_str})"
    enhanced_sql_rules = """

CRITICAL SQL GENERATION RULES (Apply to EVERY query):

1. NEVER use exact ICD code matches ALONE - ALWAYS include fuzzy search on long_title
   Example: WHERE (icd_code = '4409' OR long_title LIKE '%Atherosclerosis%')

2. For diagnosis queries, use BOTH icd_code exact match AND long_title LIKE search:
   WHERE (diagnoses_icd.icd_code = 'TARGET_CODE' AND d_icd_diagnoses.icd_version = d_icd_diagnoses.icd_version)
   OR d_icd_diagnoses.long_title LIKE '%KEYWORD%'

3. NEVER use LIMIT unless user explicitly requests "top N" or "first N"
   - "Show me all admissions" → NO LIMIT
   - "Show me the 5 most recent" → LIMIT 5

4. Use wildcards liberally for text matching:
   - 'Pneumonia' → '%Pneumonia%' OR '%pneumon%'
   - 'Atherosclerosis' → '%athero%' OR '%sclerosis%'

5. For partial text matches in medications or procedures, use:
   WHERE drug LIKE '%KEYWORD%' (not exact =)

6. When joining diagnoses_icd with d_icd_diagnoses:
   - ALWAYS join on BOTH icd_code AND icd_version
   - Include long_title in SELECT to help with interpretation

7. For "Does patient have X?" queries:
   - If result is empty, this means NO (valid negative finding)
   - Return the query anyway - empty results are meaningful

8. When checking for medications:
   - Use prescriptions table
   - Filter by subject_id AND drug LIKE '%medication%'
   - Include starttime and stoptime to show duration

EXAMPLES:

BAD (too restrictive):
SELECT * FROM diagnoses_icd WHERE icd_code = '4409' LIMIT 10

GOOD (comprehensive):
SELECT di.subject_id, di.icd_code, did.long_title
FROM diagnoses_icd di
JOIN d_icd_diagnoses did ON di.icd_code = did.icd_code AND di.icd_version = did.icd_version
WHERE di.subject_id = 10021487
  AND (di.icd_code = '4409' OR did.long_title LIKE '%Atherosclerosis%')

BAD (exact drug match):
SELECT * FROM prescriptions WHERE drug = 'Heparin'

GOOD (fuzzy drug match):
SELECT subject_id, starttime, stoptime, drug
FROM prescriptions
WHERE subject_id = 10000032 AND drug LIKE '%Heparin%'
"""

    effective_question += enhanced_sql_rules
    print(f"   [EHR] Generating SQL with {len(enhanced_sql_rules.split('\\n'))} global rules applied.")

    raw_sql = generate_sql(
        query=effective_question,
        schema_context_str=state['schema_context'],
        metrics_list=state['metrics_context'],
        concepts_list=state['concepts_context'],
        config_path='config' 
    )
    
    clean_sql = sanitize_sql(raw_sql)
    
    logic_match = None
    import re
    logic_pattern = r'\[Logic\]:\s*(.+?)(?:\n|$)'
    match = re.search(logic_pattern, raw_sql, re.IGNORECASE | re.DOTALL)
    if match:
        logic_match = match.group(1).strip()
        print(f"[Logic]: {logic_match}")
    
    print(f"\n[Draft SQL]:\n{clean_sql}\n")
    
    return {"sql_query": clean_sql}

def execute_node(state: AgentState):
    """
    Executes SQL query against MIMIC-III database.
    
    Returns results as list of dictionaries or captures
    execution error for correction workflow.
    """
    print("   [EHR] Executing SQL...")
    sql = state['sql_query']
    try:
        df = execute_query(DB_PATH, sql)
        result_dict = df.to_dict(orient='records')
        return {"query_result": result_dict, "error": None}
    except Exception as e:
        return {"query_result": None, "error": str(e)}

def correct_node(state: AgentState):
    """
    Repairs syntax errors in SQL queries.
    
    Fixes common issues: ambiguous columns, table name typos,
    JOIN errors, and missing table prefixes.
    """
    error_msg = state['error']
    print(f"   [EHR] SQL Crashed: {error_msg}")
    
    prompt = f"""
You are a SQL Repair Agent.

QUESTION: {state['question']}
FAILED SQL: {state['sql_query']}
ERROR: {error_msg}
SCHEMA: {state['schema_context']}

COMMON FIXES:
1. If "no such column" → Check table.column syntax
2. If "ambiguous column" → Add table prefix to all columns
3. If "no such table" → Verify table name spelling
4. If JOIN fails → Check foreign key relationships

Fix the SQL and return ONLY the corrected SQL query.
"""
    response = llm_tool.invoke(prompt)
    fixed_sql = sanitize_sql(response.content)
    
    print(f"\n[Fixed SQL]:\n{fixed_sql}\n")
    
    return {
        "sql_query": fixed_sql, 
        "retries": state['retries'] + 1
    }

def investigate_node(state: AgentState):
    print("   [EHR] 0 Rows. Investigating values...")
    
    plan_prompt = f"""
The following SQL returned 0 results. This could mean:
1. The queried data genuinely doesn't exist (valid NEGATIVE result)
2. There's a string mismatch (typo, case sensitivity, partial match needed)

SQL: {state['sql_query']}

YOUR TASK:
1. Identify the Table, Column, and Value that might be causing the mismatch
2. Suggest 2-3 alternative search terms (partial matches, common typos, related terms)

Format your response as:
TABLE | COLUMN | ORIGINAL_VALUE | ALTERNATIVE1, ALTERNATIVE2
"""
    plan = llm_tool.invoke(plan_prompt).content.strip()
    
    try:
        parts = [x.strip() for x in plan.split('|')]
        if len(parts) < 3:
            print("Investigation inconclusive - data may genuinely not exist")
            return {
                "investigated": True,
                "query_result": []
            }
        
        t, c, v = parts[0], parts[1], parts[2]
        alternatives = parts[3].split(',') if len(parts) > 3 else []
        
        actual_values = lookup_values_in_db(DB_PATH, t, c, v)
        print(f"Found values: {actual_values}")
        
        if not actual_values and alternatives:
            for alt in alternatives[:2]:    
                alt_clean = alt.strip()
                actual_values = lookup_values_in_db(DB_PATH, t, c, alt_clean)
                if actual_values:
                    print(f"Match found with alternative: '{alt_clean}'")
                    v = alt_clean
                    break
        
        if actual_values:
            fix_prompt = f"""
    You are a SQL repair specialist.

    Original SQL (returned 0 rows): 
    {state['sql_query']}

    Mismatched Value: '{v}'
    Actual DB Values Found: {actual_values}

    TASK: Rewrite the SQL query to use the closest matching value from the Actual DB Values list.

    CRITICAL RULES:
    1. Return ONLY valid SQL - no explanations, no markdown, no preamble
    2. Use LIKE with wildcards for fuzzy matching
    3. Pick the most relevant value from the list

    CORRECTED SQL:
    """
        response = llm_tool.invoke(fix_prompt)
        new_sql_raw = response.content
        
        sql_match = re.search(r'(SELECT\s+.+)', new_sql_raw, re.IGNORECASE | re.DOTALL)
        if sql_match:
            new_sql = sanitize_sql(sql_match.group(1))
        else:
            print("Could not extract SQL from investigation response")
            return {"investigated": True}
        
        print(f"\n   [Investigated SQL]:\n{new_sql}\n")
        return {"sql_query": new_sql, "investigated": True}
        
    except Exception as e:
        print(f"Investigation error: {e}")
        return {"investigated": True}

# Build Workflow Graph
def should_continue(state: AgentState):
    if state['error']:
        if state['retries'] >= 3: return "end"
        return "retry_error"
    
    if state['query_result'] is not None and len(state['query_result']) == 0:
        if not state['investigated']:
            return "investigate_empty"
            
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("execute", execute_node)
workflow.add_node("correct", correct_node)
workflow.add_node("investigate", investigate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "execute")

workflow.add_conditional_edges(
    "execute",
    should_continue,
    {"end": END, "retry_error": "correct", "investigate_empty": "investigate"}
)

workflow.add_edge("correct", "execute")
workflow.add_edge("investigate", "execute")

ehr_graph = workflow.compile()