import numpy as np
import yaml
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# Embedding dimension for the Gemini embedding model
GEMINI_EMBEDDING_DIM = 3072 

class HybridRetriever:
    """
    Hybrid dense + sparse retriever combining BM25 and Gemini embeddings.
    """

    def __init__(self, graph, config_path='config'):
        self.graph = graph
        
        # Two embedding models: one for indexing, one for queries
        self.doc_embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="retrieval_document"
        )

        self.query_embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="retrieval_query" 
        )
        
        self.docs = []
        self.ids = []
        self._load_configs(config_path)
        self._build_documents()
        tokenized_corpus = [self._tokenize(doc) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
 
        # Precompute dense embeddings for the corpus
        self.dense_matrix = self._get_gemini_embeddings(self.docs)

    def _get_gemini_embeddings(self, texts):
        """
        Generate embeddings for a list of documents. Uses batching and fallbacks.
        """
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = [t.replace("\n", " ") for t in batch]
            try:
                batch_embeddings = self.doc_embedding_model.embed_documents(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error during embedding generation: {e}")
                # Fallback: zero vector of expected size
                embeddings.extend([ [0.0] * GEMINI_EMBEDDING_DIM for _ in batch ])
        return np.array(embeddings)
        
    def _load_configs(self, path):
        """Load metrics and concepts YAML files."""
        with open(f"{path}/metrics.yaml") as f: 
            self.metrics = yaml.safe_load(f)

        with open(f"{path}/concepts.yaml") as f: 
            self.concepts = yaml.safe_load(f)

    def _tokenize(self, text):
        return str(text).lower().split()

    def _build_documents(self):
        """Construct document corpus from graph nodes, metrics, and concepts."""

        # Tables & Columns
        for n, attrs in self.graph.nodes(data=True):
            if attrs.get('kind') == 'table':
                self.docs.append(f"table {n}")
                self.ids.append(n)
            elif attrs.get('kind') == 'column':
                text = f"column {n} {attrs.get('dtype','')}"
                if 'long_title' in n: text += " definition description"
                self.docs.append(text)
                self.ids.append(n)
        
        # Metrics
        for m_name, m_data in self.metrics.items():
            self.docs.append(f"metric {m_name} {m_data['description']}")
            self.ids.append(f"METRIC:{m_name}")

        # Concepts
        for c_name, c_data in self.concepts.items():
            self.docs.append(f"concept {c_name} {c_data['description']}")
            self.ids.append(f"CONCEPT:{c_name}")

    def retrieve(self, query, top_k=15, alpha=0.5):
        """
        Perform hybrid retrieval using BM25 + Gemini dense embeddings.
        """
        tokenized_query = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        if bm25_scores.max() > 0: bm25_scores /= bm25_scores.max()
        
        query_vec = self.query_embedding_model.embed_query(query.replace("\n", " "))
        query_vec = np.array(query_vec).reshape(1, -1)
        dense_scores = cosine_similarity(query_vec, self.dense_matrix).ravel()
        
        hybrid_scores = (alpha * dense_scores) + ((1 - alpha) * bm25_scores)
        top_indices = hybrid_scores.argsort()[::-1][:top_k]
        
        relevant_nodes = set()
        metrics_found = []
        concepts_found = []

        for idx in top_indices:
            ident = self.ids[idx]
            if ident.startswith("METRIC:"):
                metrics_found.append(ident.split(":")[1])
            elif ident.startswith("CONCEPT:"):
                concepts_found.append(ident.split(":")[1])
            else:
                relevant_nodes.add(ident)
                if ident in self.graph:
                    relevant_nodes.update(self.graph.neighbors(ident))
        
        return self.graph.subgraph(relevant_nodes), metrics_found, concepts_found