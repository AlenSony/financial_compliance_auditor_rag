import json
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# 1. Define the exact structure we want the AI to return
class AuditReport(BaseModel):
    company_name: str = Field(description="The explicit name of the organization or company.")
    total_assets: str = Field(description="The total value of assets listed, including currency units.")
    total_liabilities: str = Field(description="The total liabilities or debts listed.")
    net_income_or_equity: str = Field(description="The net income, net position, or total equity value.")
    risk_assessment: str = Field(description="A brief 1-sentence compliance risk assessment based on the numbers.")

def run_compliance_audit(query: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory="audit_db", embedding_function=embeddings)
    
    print(f"[1/4] Searching database for context regarding: '{query}'...")
    matching_nodes = db.similarity_search(query, k=2)
    
    if not matching_nodes:
        print("[-] Error: No document layout context could be extracted from audit_db.")
        return None
        
    context_data = "\n---\n".join([node.page_content for node in matching_nodes])
    print(f"[2/4] Retrieved {len(matching_nodes)} context fragments from your PDF.")

    # 2. Set up ChatOllama with strict JSON format constraints
    print("[3/4] Initializing local model inference layer...")
    llm = ChatOllama(
        model="llama3.1", 
        temperature=0.0,
        format="json"  # Enforces raw JSON at the system level
    )
    
    system_prompt = (
        f"You are an expert financial compliance auditor. Analyze the document snippets provided below "
        f"and extract the metrics to build a JSON object matching this schema:\n"
        f"{json.dumps(AuditReport.model_json_schema(), indent=2)}\n\n"
        f"Financial Document Snippets:\n{context_data}\n\n"
        f"Return ONLY a valid, raw JSON object. Do not wrap it in markdown block tags like ```json."
    )
    
    print("[4/4] Executing structured inference extraction...")
    response = llm.invoke(system_prompt)
    
    try:
        # Parse the output text directly into our Pydantic structural validator
        cleaned_content = response.content.strip()
        parsed_json = json.loads(cleaned_content)
        validated_report = AuditReport(**parsed_json)
        return validated_report
    except Exception as e:
        print(f"[-] Validation/Parsing Error: {e}")
        print(f"Raw Model Output was:\n{response.content}")
        return None

if __name__ == "__main__":
    audit_summary = run_compliance_audit("Extract the balance sheet summary figures, total assets, and liabilities.")
    
    print("\n================ OFFICIAL AUDIT REPORT ================")
    if audit_summary:
        print(audit_summary.model_dump_json(indent=4))
    else:
        print("[-] Audit compilation failed or returned empty.")
    print("=======================================================")