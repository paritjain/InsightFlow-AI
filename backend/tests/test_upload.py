import os
import sys
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from main import app

def test_multi_file_upload():
    client = TestClient(app)
    file1 = os.path.join(BASE_DIR, "test_datasets", "test_sales.xlsx")
    file2 = os.path.join(BASE_DIR, "test_datasets", "test_customers.xlsx")

    if not os.path.exists(file1) or not os.path.exists(file2):
        return

    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        files = [
            ("files", ("test_sales.xlsx", f1.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
            ("files", ("test_customers.xlsx", f2.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
        ]
        resp = client.post("/api/projects/upload", files=files)
        assert resp.status_code == 200
        res_data = resp.json()
        assert res_data.get("status") == "success"
        assert "project_id" in res_data

