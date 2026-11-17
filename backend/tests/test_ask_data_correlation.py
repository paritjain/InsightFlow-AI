import unittest
import pandas as pd
import numpy as np
from engine.ask_data import ask_your_data

class TestAskDataCorrelation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Synthetic test dataset containing all 9 requested variables plus dimension & date
        np.random.seed(42)
        n = 100
        marketing = np.random.uniform(1000, 5000, n)
        leads = marketing * 0.5 + np.random.normal(0, 100, n)
        customers = leads * 0.2 + np.random.normal(0, 10, n)
        revenue = customers * 500 + np.random.normal(0, 1000, n)
        operating_cost = revenue * 0.4 + np.random.normal(0, 500, n)
        profit = revenue - operating_cost
        returns = np.random.uniform(5, 50, n) - profit * 0.001
        satisfaction = np.random.uniform(1, 5, n)
        delivery_hours = np.random.uniform(12, 72, n)

        cls.df = pd.DataFrame({
            'Marketing_Spend': marketing,
            'Leads': leads,
            'Customers_Acquired': customers,
            'Revenue': revenue,
            'Operating_Cost': operating_cost,
            'Profit': profit,
            'Returns': returns,
            'Customer_Satisfaction': satisfaction,
            'Delivery_Hours': delivery_hours,
            'Sales_Channel': np.random.choice(['Online', 'Retail', 'Direct'], n),
            'Booking_Date': pd.date_range(start='2026-01-01', periods=n, freq='D')
        })

    def test_01_single_pair_pearson(self):
        query = "Calculate the Pearson correlation between Revenue and Profit."
        res = ask_your_data(self.df, query)
        plan = res["query_plan"]
        self.assertIn(plan["intent"], ["relationship_single", "relationship"])
        self.assertEqual(plan["aggregation"], None)
        self.assertEqual(plan["dimension"], None)
        self.assertIn("Pearson r =", res["answer"])
        self.assertNotIn("Spearman r =", res["answer"])

    def test_02_single_pair_revenue_operating_cost(self):
        query = "Calculate the Pearson correlation between Revenue and Operating_Cost."
        res = ask_your_data(self.df, query)
        plan = res["query_plan"]
        self.assertIn(plan["intent"], ["relationship_single", "relationship"])
        self.assertEqual(plan["aggregation"], None)
        self.assertEqual(plan["dimension"], None)
        self.assertIn("Pearson r =", res["answer"])

    def test_03_multiple_explicit_pairs(self):
        query = "Compare exactly these three Pearson correlations: Revenue vs Profit, Revenue vs Operating_Cost, and Profit vs Operating_Cost."
        res = ask_your_data(self.df, query)
        plan = res["query_plan"]
        self.assertEqual(plan["intent"], "relationship_multiple")
        self.assertEqual(len(plan["pairs"]), 3)
        self.assertEqual(plan["aggregation"], None)
        self.assertEqual(plan["dimension"], None)
        self.assertIn("Revenue vs Profit", res["answer"])
        self.assertIn("Revenue vs Operating_Cost", res["answer"])
        self.assertIn("Profit vs Operating_Cost", res["answer"])

    def test_04_correlation_matrix_9_vars(self):
        query = "Create a Pearson correlation matrix using Marketing_Spend, Leads, Customers_Acquired, Revenue, Operating_Cost, Profit, Returns, Customer_Satisfaction, and Delivery_Hours."
        res = ask_your_data(self.df, query)
        plan = res["query_plan"]
        self.assertEqual(plan["intent"], "correlation_matrix")
        self.assertEqual(len(plan["variables"]), 9)
        self.assertEqual(plan["aggregation"], None)
        self.assertEqual(plan["dimension"], None)
        self.assertEqual(res["chart_type"], "CORRELATION_HEATMAP")
        # 9x9 matrix = 81 data points
        self.assertEqual(len(res["chart_data"]), 81)
        self.assertIn("Calculated complete Pearson correlation matrix across 9 variables", res["answer"])

    def test_05_strongest_weakest_ranking(self):
        query = "Identify the strongest positive and strongest negative Pearson correlations among Marketing_Spend, Leads, Customers_Acquired, Revenue, Operating_Cost, Profit, Returns, Customer_Satisfaction, and Delivery_Hours."
        res = ask_your_data(self.df, query)
        plan = res["query_plan"]
        self.assertEqual(plan["intent"], "correlation_ranking")
        self.assertEqual(len(plan["variables"]), 9)
        self.assertEqual(plan["aggregation"], None)
        self.assertEqual(plan["dimension"], None)
        self.assertIn("Strongest Positive Correlation", res["answer"])
        self.assertIn("Strongest Negative Correlation", res["answer"])
        self.assertIn("Correlation indicates linear co-variance and does not imply direct causation", res["answer"])

    def test_06_pearson_and_spearman(self):
        query = "Calculate both Pearson and Spearman correlations between Revenue and Operating_Cost."
        res = ask_your_data(self.df, query)
        plan = res["query_plan"]
        self.assertIn("pearson", plan["methods"])
        self.assertIn("spearman", plan["methods"])
        self.assertIn("Pearson r =", res["answer"])
        self.assertIn("Spearman r =", res["answer"])

    def test_07_no_grouping_no_aggregation(self):
        query = "Calculate the correlation between Revenue and Operating_Cost with no grouping and no aggregation."
        res = ask_your_data(self.df, query)
        plan = res["query_plan"]
        self.assertEqual(plan["aggregation"], None)
        self.assertEqual(plan["dimension"], None)

    def test_08_backward_compatibility_aggregations(self):
        res_tot = ask_your_data(self.df, "What is the total revenue?")
        self.assertEqual(res_tot["query_plan"]["intent"], "aggregate")
        self.assertEqual(res_tot["query_plan"]["aggregation"], "sum")

        res_by_dim = ask_your_data(self.df, "What is revenue by Sales_Channel?")
        self.assertIn(res_by_dim["query_plan"]["intent"], ["group_comparison", "ranking"])
        self.assertEqual(res_by_dim["query_plan"]["dimension"], "Sales_Channel")
        self.assertEqual(res_by_dim["query_plan"]["dimension"], "Sales_Channel")

    def test_09_exact_acceptance_test_query(self):
        query = (
            "Create a Pearson correlation matrix using Marketing_Spend, Leads, Customers_Acquired, "
            "Revenue, Operating_Cost, Profit, Returns, Customer_Satisfaction, and Delivery_Hours. "
            "Calculate correlations using raw row-level observations with no aggregation and no grouping by any dimension. "
            "Return the complete matrix with variable names as both rows and columns, including diagonal values. "
            "Do not analyze only one pair and do not omit any variable."
        )
        res = ask_your_data(self.df, query)
        plan = res["query_plan"]
        self.assertEqual(plan["intent"], "correlation_matrix")
        self.assertEqual(plan["methods"], ["pearson"])
        self.assertEqual(plan["aggregation"], None)
        self.assertEqual(plan["dimension"], None)
        self.assertEqual(plan["granularity"], "row-level")
        self.assertEqual(len(plan["variables"]), 9)
        self.assertEqual(len(res["chart_data"]), 81)
        self.assertIn("Calculated complete Pearson correlation matrix across 9 variables", res["answer"])
        self.assertNotIn("Spearman", res["answer"])

    def test_10_exact_ranking_acceptance_test_query(self):
        query = (
            "Among Marketing_Spend, Leads, Customers_Acquired, Revenue, Operating_Cost, Profit, Returns, Customer_Satisfaction, and Delivery_Hours, "
            "identify the three strongest positive Pearson correlations and the three strongest negative Pearson correlations. "
            "Calculate all correlations using raw row-level observations with no aggregation or grouping. "
            "Return the exact variable pairs and correlation coefficients, rank them from strongest to weakest, and do not include Spearman correlations or substitute any other variables."
        )
        res = ask_your_data(self.df, query)
        plan = res["query_plan"]
        self.assertEqual(plan["intent"], "correlation_ranking")
        self.assertEqual(plan["methods"], ["pearson"])
        self.assertEqual(plan["aggregation"], None)
        self.assertEqual(plan["dimension"], None)
        self.assertEqual(plan["granularity"], "row-level")
        self.assertEqual(len(plan["variables"]), 9)
        self.assertIn("36", str(res["evidence"]))
        self.assertIn("TOP 3 STRONGEST POSITIVE PEARSON CORRELATIONS", res["answer"])
        self.assertIn("TOP 3 STRONGEST NEGATIVE PEARSON CORRELATIONS", res["answer"])
        self.assertNotIn("Spearman", res["answer"])
        self.assertNotIn("Sales_Channel", str(plan))

if __name__ == '__main__':
    unittest.main()
