import os
import pytest
from fastapi.testclient import TestClient
import sys

try:
    from backend.main import app
    from backend.engine.pdf_engine import DocumentIntelligenceEngine
    from backend.agents.tools import PDFIntelligenceTool
    from backend.agents.graph import LangGraphDataAgent
except ModuleNotFoundError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    from main import app
    from engine.pdf_engine import DocumentIntelligenceEngine
    from agents.tools import PDFIntelligenceTool
    from agents.graph import LangGraphDataAgent

TEST_PDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_financial_report.pdf"))

def test_01_pdf_engine_extraction():
    """Verify DocumentIntelligenceEngine extracts text, sections, and tables accurately."""
    assert os.path.exists(TEST_PDF_PATH), f"Test PDF does not exist at {TEST_PDF_PATH}"
    engine = DocumentIntelligenceEngine()
    doc_info = engine.extract_document(TEST_PDF_PATH)
    
    assert doc_info["filename"] == "test_financial_report.pdf"
    assert doc_info["total_pages"] >= 2
    assert len(doc_info["passages"]) >= 2
    assert "tables" in doc_info
    
    # Check that financial content was captured
    full_text = " ".join(doc_info["text_by_page"].values())
    assert "12,450,000" in full_text
    assert "North America" in full_text
    assert "EBITDA" in full_text

def test_02_pdf_engine_factual_retrieval():
    """Verify document QA answers factual questions correctly and cites pages."""
    engine = DocumentIntelligenceEngine()
    engine.extract_document(TEST_PDF_PATH)
    
    # Query 1: Total Revenue
    res_rev = engine.answer_question("What was the total revenue in Q4?")
    assert res_rev["status"] == "PASS"
    assert "12,450,000" in res_rev["answer"] or "12.45" in res_rev["answer"]
    assert len(res_rev["page_references"]) > 0
    assert len(res_rev["evidence"]) > 0
    
    # Query 2: Regional breakdown from table
    res_table = engine.answer_question("What was the revenue and growth for North America?")
    assert res_table["status"] == "PASS"
    assert "5,200,000" in res_table["answer"]
    
    # Query 3: Executive Summary
    res_sum = engine.answer_question("Provide an executive summary of this report")
    assert res_sum["status"] == "PASS"
    assert "Executive Summary" in res_sum["answer"] or "Summary" in res_sum["answer"]

def test_03_pdf_engine_zero_hallucination():
    """Verify that ungrounded questions do not hallucinate facts."""
    engine = DocumentIntelligenceEngine()
    engine.extract_document(TEST_PDF_PATH)
    
    res_unknown = engine.answer_question("What is the CEO's favorite movie and pet dog name?")
    # Answer should explicitly state the document does not contain this information
    answer_lower = res_unknown["answer"].lower()
    assert "does not contain" in answer_lower or "not mentioned" in answer_lower or "could not be verified" in answer_lower

def test_04_pdf_intelligence_tool():
    """Verify PDFIntelligenceTool runs with engine telemetry."""
    engine = DocumentIntelligenceEngine()
    engine.extract_document(TEST_PDF_PATH)
    tool = PDFIntelligenceTool(engine)
    
    result = tool.run(query="What was the EBITDA margin in Q4?")
    assert result["status"] == "PASS"
    assert "25.7%" in result["answer"] or "25.7" in result["answer"] or "3,200,000" in result["answer"]
    assert result["source_file"] == "test_financial_report.pdf"
    assert result["passages_retrieved"] > 0

def test_05_langgraph_agent_pdf_routing():
    """Verify LangGraphDataAgent routes document questions to doc_qa_node and formats ChatGPT-style markdown."""
    engine = DocumentIntelligenceEngine()
    engine.extract_document(TEST_PDF_PATH)
    
    agent = LangGraphDataAgent(dfs_dict={}, pdf_documents={"test_financial_report.pdf": engine})
    
    response = agent.process_message("Summarize the key financial findings from the uploaded PDF document")
    
    assert response["tool_used"] == "DocumentIntelligenceTool"
    assert response["intent"] == "DOCUMENT_QA"
    assert response["status"] == "PASS"
    assert len(response["answer"]) > 50
    # Check ChatGPT-style markdown formatting
    assert "#" in response["answer"] or "**" in response["answer"] or "-" in response["answer"]
    assert response["source_file"] == "test_financial_report.pdf"

def test_06_fastapi_upload_and_chat_pdf():
    """Verify end-to-end FastAPI upload of PDF and /chat endpoint execution."""
    client = TestClient(app)
    
    with open(TEST_PDF_PATH, "rb") as f:
        upload_resp = client.post(
            "/api/projects/upload",
            files={"files": ("test_financial_report.pdf", f, "application/pdf")}
        )
    assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
    upload_data = upload_resp.json()
    project_id = upload_data["project_id"]
    
    # Now ask a question about the uploaded PDF
    chat_resp = client.post(
        f"/api/project/{project_id}/chat",
        json={"message": "What is the total revenue reported in Q4 and what was the EBITDA?"}
    )
    assert chat_resp.status_code == 200, f"Chat failed: {chat_resp.text}"
    chat_data = chat_resp.json()
    
    assert chat_data["status"] == "PASS"
    assert chat_data["tool_used"] == "DocumentIntelligenceTool"
    assert "12,450,000" in chat_data["answer"] or "12.45" in chat_data["answer"]
    assert chat_data["source_file"] == "test_financial_report.pdf"
    assert len(chat_data.get("page_references", [])) > 0

def test_07_test_pdf_hello_world():
    """Verify that uploading test.pdf correctly returns its exact contents and grounded citations."""
    client = TestClient(app)
    test_pdf_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test.pdf"))
    assert os.path.exists(test_pdf_file)
    
    with open(test_pdf_file, "rb") as f:
        upload_resp = client.post(
            "/api/projects/upload",
            files={"files": ("test.pdf", f, "application/pdf")}
        )
    assert upload_resp.status_code == 200
    project_id = upload_resp.json()["project_id"]
    
    chat_resp = client.post(
        f"/api/project/{project_id}/chat",
        json={"message": "What is written in test.pdf?"}
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["status"] == "PASS"
    assert chat_data["tool_used"] == "DocumentIntelligenceTool"
    assert "Hello world" in chat_data["answer"]
    assert chat_data["source_file"] == "test.pdf"
    assert 1 in chat_data.get("page_references", [])
