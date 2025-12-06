# Clinical Intelligence Assistant

**Master of Science in Applied Data Science Capstone Project**  
**University of Chicago**  
**Team:** Savita Moorthi, Yeochan Youn, Gyujin Seo, Aadya Nair

---

## Project Overview

The **Clinical Intelligence Assistant** is an advanced multi-agent AI system designed to answer complex clinical queries by orchestrating specialized retrieval and generation workflows across structured and unstructured electronic health record (EHR) data. This system leverages state-of-the-art LLMs, vector databases, and agentic AI architectures to provide evidence-based clinical insights.

### Key Features

- **Multi-Agent Architecture**: Orchestrated workflow using LangGraph with three specialized agents:
  - **EHR Agent**: Text-to-SQL system for querying structured MIMIC-III database
  - **Reports Agent**: Vector-based retrieval for unstructured clinical documents (radiology reports)
  - **Supervisor Agent**: Intelligent coordinator managing agent collaboration and response synthesis

- **Hybrid Retrieval System**: Combines graph-based schema navigation, semantic search, and cross-encoder reranking for optimal information retrieval

- **Medical Domain Optimization**: 
  - ICD code fuzzy matching
  - Medical concept mapping
  - Clinical metrics awareness
  - Patient context retention across multi-turn conversations

- **Evaluation Framework**: Comprehensive testing using RAGAS metrics (faithfulness, answer relevance, context precision)

- **Full-Stack Application**: React-based frontend with FastAPI backend for interactive clinical query interface

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Supervisor Agent    │
                  │  (LangGraph)         │
                  └──────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  EHR Agent   │  │ Reports Agent│  │  Synthesizer │
    │  (Text→SQL)  │  │ (Vector DB)  │  │    (LLM)     │
    └──────┬───────┘  └──────┬───────┘  └──────────────┘
           │                 │
           ▼                 ▼
    ┌──────────────┐  ┌──────────────┐
    │  MIMIC-III   │  │   Zilliz     │
    │   SQLite     │  │  Vector DB   │
    └──────────────┘  └──────────────┘
```

### Technology Stack

**Backend:**
- Python 3.8+
- LangGraph & LangChain (Agentic AI framework)
- FastAPI (REST API)
- Google Gemini 2.5 Flash (LLM)
- OpenAI GPT-4 (Alternative LLM)
- Sentence Transformers (Embeddings)
- Zilliz Cloud (Vector Database)
- SQLite (MIMIC-III Database)
- NetworkX (Schema Graph)
- RAGAS (Evaluation Framework)

**Frontend:**
- React 18
- Ant Design & Material-UI
- React Markdown
- Axios/HTTP Client

**Data:**
- MIMIC-III Clinical Database
- Custom medical ontology (ICD codes, metrics, concepts)

---

## Project Structure

```
Clinical_Intelligence_Assistant/
│
├── agent/                          # Multi-agent system
│   ├── supervisor.py               # Orchestrator agent
│   ├── ehr_agent.py                # Structured data (SQL) agent
│   └── reports_agent.py            # Unstructured data (RAG) agent
│
├── src/                            # Core utilities
│   ├── db_graph.py                 # Database schema graph builder
│   ├── retrieval.py                # Hybrid retrieval logic
│   ├── generation.py               # SQL generation
│   └── utils.py                    # Helper functions
│
├── config/                         # Domain knowledge
│   ├── concepts.yaml               # Medical concepts mapping
│   ├── metrics.yaml                # Clinical metrics definitions
│   └── rules.yaml                  # Query generation rules
│
├── data/                           # Data storage
│   ├── mimic.db                    # MIMIC-III SQLite database
│   └── raw/                        # Raw data files
│
├── db_loader/                      # Database utilities
│   └── db_loader.py                # Data ingestion scripts
│
├── test/                           # Evaluation suite
│   ├── eval_clinical.py            # Clinical queries evaluation
│   ├── eval_genbi.py               # General BI evaluation
│   ├── eval_hybrid.py              # Multi-agent evaluation
│   ├── clinical_reports_test.json  # Test cases
│   ├── genbi_test.json             # Test cases
│   └── hybrid_test.json            # Test cases
│
├── frontend/                       # React UI
│   ├── src/                        # React components
│   ├── public/                     # Static assets
│   └── package.json                # Frontend dependencies
│
├── main.py                         # FastAPI application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (NOT in git)
└── README.md                       # This file
```

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- OpenAI API key
- Google API key (for Gemini)
- Zilliz Cloud account (optional, for vector DB)

### Step 1: Clone the Repository

```bash
git clone https://github.com/savitamoorthi/Clinical_Intelligence_Assistant.git
cd Clinical_Intelligence_Assistant
```

### Step 2: Python Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Environment Variables

Create a `.env` file in the root directory with the following content:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Google Gemini Configuration
GOOGLE_API_KEY=your-google-api-key-here

# Zilliz Cloud Configuration (Optional)
ZILLIZ_URI=your-zilliz-uri-here
ZILLIZ_TOKEN=your-zilliz-token-here
```

### Step 4: Frontend Setup

```bash
cd frontend
npm install
cd ..
```

---

## Usage

### Running the Backend Server

```bash
# From project root directory
python main.py
```

The API server will start at: `http://localhost:8000`

**API Endpoints:**
- `GET /health` - Health check
- `POST /chat` - Submit clinical query

### Running the Frontend

```bash
# In a separate terminal
cd frontend
npm start
```

The UI will be available at: `http://localhost:3000`

### Command Line Interface

You can also interact with the supervisor agent directly:

```bash
python agent/supervisor.py
```

### Running Individual Agents

**EHR Agent (SQL):**
```bash
python agent/ehr_agent.py
```

**Reports Agent (RAG):**
```bash
python agent/reports_agent.py
```

---

## Evaluation

The project includes comprehensive evaluation scripts using RAGAS metrics:

### Running Evaluation Tests

```bash
# Clinical queries evaluation
python -m test.eval_clinical

# General BI queries evaluation
python -m test.eval_genbi

# Hybrid multi-agent evaluation
python -m test.eval_hybrid
```

### Evaluation Metrics

- **Faithfulness**: Measures answer accuracy relative to retrieved context
- **Answer Relevance**: Assesses how well the answer addresses the question
- **Context Precision**: Evaluates retrieval quality
- **Context Recall**: Measures completeness of retrieved information

Results are saved as CSV files in the `test/` directory.

---

## Sample Queries

### Structured Data (EHR Agent)
```
"What are the vital signs for patient 10006?"
"Show me lab results for hemoglobin under 10 g/dL"
"Find patients diagnosed with pneumonia in 2023"
```

### Unstructured Data (Reports Agent)
```
"What did the chest X-ray show for patient 10006?"
"Summarize the radiology report 10000032-RR-14"
"Find discharge summaries mentioning cardiac complications"
```

### Hybrid Queries (Multi-Agent)
```
"Show me labs and radiology findings for patient 10006"
"Are there any patients with abnormal cardiac enzymes and corresponding ECG reports?"
"Compare vitals and clinical notes for admission 120503"
```

---

## Security & Privacy

- **PII Masking**: Sensitive information is automatically redacted in outputs
- **Environment Variables**: API keys stored securely in `.env` (gitignored)
- **Database Access**: Local SQLite prevents unauthorized network access
- **Input Sanitization**: SQL injection protection via parameterized queries

**HIPAA Compliance Note:** This is a research prototype using the publicly available MIMIC-III dataset. Do not use with real patient data without proper authorization and security review.

---

## Development

### Adding New Medical Concepts

Edit `config/concepts.yaml`:

```yaml
new_concept:
  synonyms:
    - "alternative term 1"
    - "alternative term 2"
  icd_codes:
    - "ICD10.CODE"
```

### Configuring Metrics

Edit `config/metrics.yaml` to add clinical measurements and normal ranges.

### Extending Agent Capabilities

Each agent (`ehr_agent.py`, `reports_agent.py`, `supervisor.py`) is built using LangGraph's `StateGraph`. Add new nodes by:

1. Defining a node function
2. Adding the node to the workflow
3. Configuring edges and routing logic

## Academic Context

This project was developed as a capstone for the **Master of Science in Applied Data Science** program at the **University of Chicago**. It demonstrates advanced applications of:

- Multi-agent systems and agentic AI
- Retrieval-Augmented Generation (RAG)
- Text-to-SQL for healthcare analytics
- Vector databases and semantic search
- LLM prompt engineering and orchestration
- Clinical informatics and medical ontologies

---

## License

This is an academic project for educational purposes.

---

## Acknowledgments

- University of Chicago MSADS faculty and advisors
- MIT Laboratory for Computational Physiology (MIMIC-III dataset)
- LangChain and LangGraph communities
- Open-source contributors to all dependencies

---

## Contact

For questions or collaboration opportunities, please reach out via GitHub issues or pull requests.
Contact the team through - savitamoorthi@outlook.com

---

*Last Updated: December 2025*
