import unittest
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from fastapi.testclient import TestClient
from main import app

class TestLiveHTTPAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_list_projects(self):
        resp = self.client.get("/api/projects")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_02_get_project_details(self):
        resp = self.client.get("/api/project/proj-demo-q3-sales")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["id"], "proj-demo-q3-sales")
        self.assertIn("profile", data)
        self.assertIn("eda_results", data)
        self.assertIn("insights", data)
        self.assertIn("action_items", data)

    def test_03_ask_question(self):
        resp = self.client.post("/api/project/proj-demo-q3-sales/ask", json={"question": "Why did sales decline?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        self.assertIn("calculation_formula", data)

    def test_04_export_deliverables(self):
        excel_resp = self.client.get("/api/project/proj-demo-q3-sales/export/excel")
        self.assertEqual(excel_resp.status_code, 200)
        self.assertGreater(len(excel_resp.content), 1000)

        pdf_resp = self.client.get("/api/project/proj-demo-q3-sales/export/pdf")
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertGreater(len(pdf_resp.content), 1000)

if __name__ == "__main__":
    unittest.main()

