import unittest
import os
import sys
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from main import app

class TestInsightFlowAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_dir = os.path.join(BASE_DIR, "test_datasets")

    def test_01_list_projects(self):
        response = self.client.get("/api/projects")
        self.assertEqual(response.status_code, 200)
        projects = response.json()
        self.assertIsInstance(projects, list)
        self.assertGreater(len(projects), 0)

    def test_02_get_demo_project_details(self):
        response = self.client.get("/api/project/proj-demo-q3-sales")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], "proj-demo-q3-sales")
        self.assertIn("profile", data)
        self.assertIn("eda_results", data)
        self.assertIn("insights", data)
        self.assertIn("action_items", data)

    def test_03_file_upload_endpoint(self):
        file_path = os.path.join(self.test_dir, "test_sales.xlsx")
        with open(file_path, "rb") as f:
            files = [("files", ("test_sales.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))]
            response = self.client.post("/api/projects/upload", files=files)
        
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res["status"], "success")
        self.assertIn("project_id", res)

        proj_id = res["project_id"]
        # Fetch project details
        detail_res = self.client.get(f"/api/project/{proj_id}")
        self.assertEqual(detail_res.status_code, 200)

    def test_04_ask_endpoint(self):
        payload = {"question": "What is the total revenue?"}
        response = self.client.post("/api/project/proj-demo-q3-sales/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertIn("answer", res)
        self.assertIn("calculation_formula", res)

    def test_05_export_endpoints(self):
        excel_res = self.client.get("/api/project/proj-demo-q3-sales/export/excel")
        self.assertEqual(excel_res.status_code, 200)
        self.assertEqual(excel_res.headers["content-type"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        pdf_res = self.client.get("/api/project/proj-demo-q3-sales/export/pdf")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers["content-type"], "application/pdf")

if __name__ == "__main__":
    unittest.main()
