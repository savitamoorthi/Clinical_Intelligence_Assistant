import pandas as pd
import sqlite3

def sanitize_sql(sql: str) -> str:
    """
    Perform lightweight SQL sanitization to remove markdown code fences and ensure
    the SQL string ends with a semicolon.
    """
    if not sql:
        return ""

    clean = sql.replace("```sql", "").replace("```", "").strip()
    
    if clean.lower().startswith("sql"):
        clean = clean[3:].strip()
        
    if not clean.endswith(";"):
        clean += ";"
        
    return clean

def lookup_values_in_db(db_path: str, table: str, column: str, search_term: str):
    """
    Search a SQLite database column for values similar to the provided search term.

    Returns:
        list of matching values or a single-element list with an error string.
    """
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} LIKE ? LIMIT 10"
        cursor = conn.cursor()
        cursor.execute(query, (f'%{search_term}%',))
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        return [f"Error looking up values: {str(e)}"]

def mask_pii(df: pd.DataFrame, config_path='config') -> pd.DataFrame:
    """
    Placeholder for PII masking. Currently returns the original DataFrame.

    This function intentionally does *not* include masking logic to avoid
    publishing internal configurations or rules.
    """
    return df

def execute_query(db_path: str, sql: str) -> pd.DataFrame:
    """
    Execute a SQL query on a SQLite database and return the result as a DataFrame.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        raise e 
    finally:
        conn.close()