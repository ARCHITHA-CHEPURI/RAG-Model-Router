import unittest

from llm.llm_router import LLMRouter
from llm.router_strategies import BaseRouter, RuleRouter


class RouterStrategyTests(unittest.TestCase):
    def test_base_router_always_uses_large_model(self):
        self.assertEqual(BaseRouter().choose_model("Hello"), ("large", None))

    def test_rule_router_routes_simple_and_complex_queries(self):
        router = RuleRouter()

        self.assertEqual(router.choose_model("Hello"), ("small", None))
        self.assertEqual(
            router.choose_model("Write a complex React component"),
            ("large", None),
        )

    def test_ml_router_uses_the_same_route_contract(self):
        model, confidence = LLMRouter().choose_model("What is RAG?")

        self.assertIn(model, {"small", "large"})
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_strategies_reject_empty_queries(self):
        for router in (BaseRouter(), RuleRouter(), LLMRouter()):
            with self.subTest(router=type(router).__name__):
                with self.assertRaises(ValueError):
                    router.choose_model("  ")


if __name__ == "__main__":
    unittest.main()
