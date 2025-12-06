from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import yaml

class SQLResponse(BaseModel):
    """Structured response schema for SQL generation."""
    sql_query: str = Field(
        description="The raw SQLite query. Do not use markdown formatting."
    )
    explanation: str = Field(
        description="A brief logic explanation of how the query answers the question."
    )

def generate_sql(query, schema_context_str, metrics_list, concepts_list, config_path='config'):
    """
    Generate SQL using Gemini via LangChain structured output, guided by rules,
    metrics, and concept definitions.
    """
    
    # Load Rules 
    with open(f"{config_path}/rules.yaml") as f:
        rules_config = yaml.safe_load(f)
    
    global_instrs = rules_config.get('global_instructions', [])
    mappings = rules_config.get('column_mappings', {})
    mapping_text_list = [f"- {key}: {instruction}" for key, instruction in mappings.items()]

    # Load Metrics 
    with open(f"{config_path}/metrics.yaml") as f: 
        all_metrics = yaml.safe_load(f)

    metric_instr = [f"Metric '{m}': {all_metrics[m]['sql_template']}" 
                    for m in metrics_list 
                    if m in all_metrics]

    # Load Concepts 
    with open(f"{config_path}/concepts.yaml") as f:
        all_concepts = yaml.safe_load(f)
    concept_instr = []
    for c in concepts_list:
        if c in all_concepts:
            data = all_concepts[c]
            if 'filter_logic' in data:
                concept_instr.append(f"Concept '{c}': Use filter ({data['filter_logic']})")
            if 'usage_guide' in data:
                concept_instr.append(f"*** RULE FOR '{c}' ***\n{data['usage_guide']}")

    # Build Prompt 
    prompt = f"""
    You are a Healthcare SQL Expert.
    
    QUESTION: {query}
    
    SCHEMA CONTEXT:
    {schema_context_str}
    
    BUSINESS RULES (Applied to this specific question):
    {chr(10).join(metric_instr)}
    {chr(10).join(concept_instr)}
    
    GLOBAL COLUMN MAPPINGS (Memorize These):
    {chr(10).join(mapping_text_list)}
    
    CRITICAL INSTRUCTIONS:
    {chr(10).join([f"{i+1}. {rule}" for i, rule in enumerate(global_instrs)])}
    
    Return valid SQLite SQL compatible with the schema provided.
    """
    
    # Invoke LLM 
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(SQLResponse)
    
    print(f"   [EHR] 🧠 Generating SQL with {len(mapping_text_list)} global rules applied.")
    result = structured_llm.invoke(prompt)
    
    # Normalize structured output
    if isinstance(result, dict):
        explanation = result.get("explanation", "")
        sql_query = result.get("sql_query", "")
    else:
        explanation = getattr(result, "explanation", "")
        sql_query = getattr(result, "sql_query", "")
    
    print(f"      [Logic]: {explanation}")
    return sql_query