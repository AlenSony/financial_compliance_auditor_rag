# Local Financial & Compliance Auditor (Phase 2 RAG)

A secure, air-gapped Retrieval-Augmented Generation (RAG) system engineered to parse multi-column corporate financial sheets, execute layout-aware semantic lookups, and enforce deterministic, machine-readable JSON compliance reporting.

This project addresses two core challenges in production AI engineering: preserving document structural relationships (like financial tables) during data chunking, and forcing non-deterministic LLMs to output validated, structured data payloads.

---

## 🛠️ System Architecture

The pipeline processes document layouts and enforces structural validation through two decoupled modules:

1. **Layout-Aware Ingestion Pipeline (`pdf_parser.py`):** Utilizes structural parsing to process tabular inputs page-by-page. Content is divided using optimized character windows ($1500$ character size with a $300$ character overlap) to keep table rows structurally sound. Text segments are transformed using the `nomic-embed-text` vector model and persisted locally within an SQLite-backed Chroma database (`audit_db/`).
2. **Structured Compliance Engine (`compliance_agent.py`):** Executes high-dimensional vector lookups across financial matrices ($k=2$). Extracted nodes are fed into a localized Llama 3.1 inference layer. Rather than returning raw conversational text, the model output is constrained by a strict **Pydantic Schema** using native JSON format controls, generating a clean, machine-readable audit block.

---

## 🧰 Technical Stack

* **LLM Engine:** [Ollama](https://ollama.com/) running **Llama 3.1 (8B)** (`temperature=0.0`)
* **Embedding Model:** `nomic-embed-text` (Dedicated 274 MB high-performance embedding engine)
* **Framework Layer:** LangChain Core, `langchain-ollama`, and `langchain-chroma`
* **Data Validation:** Pydantic v2
* **Storage Matrix:** ChromaDB Engine

---

## 🚀 Local Installation & Deployment

### 1. Initialize the Local Model Registry
Ensure Ollama is running in the background, then pull the required reasoning and embedding weights:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text

2. Environment Activation & Dependencies
Navigate to the project root directory and execute the following bootstrap steps:

Bash
# Initialize Python virtual sandbox
python -m venv venv

# Activate the sandbox workspace (Windows CMD)
.\venv\Scripts\activate.bat

# Install required framework libraries
pip install langchain-community langchain-chroma langchain-ollama pydantic pypdf
3. Hydrate Target Document Repository
Place your target financial statement or balance sheet PDF inside the tracking folder:

Plaintext
financial_compliance_auditor/
└── target_documents/
    └── statement.pdf
🏃‍♂️ Execution Matrix
Step 1: Execute Document Indexing
Process the target document layout and vectorize its content to local disk storage:

Bash
python pdf_parser.py
Expected Output: Logs confirmation of page read lengths, chunk distribution counts, and vector store indexing status.

Step 2: Generate the Structural Compliance Report
Run the audit engine to analyze the vectorized data and output structured, validated results:

Bash
python compliance_agent.py
Production JSON Audit Return Sample:
JSON
{
    "company_name": "XYZ, Inc.",
    "total_assets": "$6,858,029",
    "total_liabilities": "$2,887,230",
    "net_income_or_equity": "$3,970,799",
    "risk_assessment": "The company has a moderate risk of financial instability due to its high total liabilities relative to total equity assets."
}"# financial_compliance_auditor_rag" 
