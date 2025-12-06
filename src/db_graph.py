import sqlite3
import networkx as nx

def build_schema_graph(db_path: str):
    """
    Build a schema graph from a SQLite database.

    Nodes:
        - Table nodes   (kind="table")
        - Column nodes  (kind="column", dtype=<sqlite_type>)

    Edges:
        - Table → Column edges (rel="has_column")
        - Foreign key column relationships (rel="fk")

    Returns:
        networkx.Graph representing tables, columns, and FK relationships.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    
    G = nx.Graph()
    
    for t in tables:
        G.add_node(t, kind='table')
        
        # Get Columns
        cur.execute(f"PRAGMA table_info({t})")
        columns = cur.fetchall()
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            node_name = f"{t}.{col_name}"
            # Store type to help LLM know if it's a string or number
            G.add_node(node_name, kind='column', dtype=col_type)
            G.add_edge(t, node_name, rel='has_column')
            
        # Get Foreign Keys (Crucial for auto-joining)
        cur.execute(f"PRAGMA foreign_key_list({t})")
        fks = cur.fetchall()
        for fk in fks:
            ref_table = fk[2]
            from_col = f"{t}.{fk[3]}"
            to_col = f"{ref_table}.{fk[4]}"
            # Add edge between columns representing the join
            G.add_edge(from_col, to_col, rel='fk')
            
    conn.close()
    return G