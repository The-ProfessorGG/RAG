import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_core import AdvancedRAG, rewrite_query


class AdvancedRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = AdvancedRAG(ROOT / "documents", top_k=4, candidate_k=10)

    def test_loads_synthetic_corpus(self):
        self.assertEqual(len(self.rag.documents), 8)
        self.assertGreaterEqual(len(self.rag.chunks), 24)

    def test_rewrite_expands_leave_question(self):
        rewritten = rewrite_query("Can unused leave roll into next year?")
        self.assertIn("carry-over", rewritten)
        self.assertIn("deadline", rewritten)

    def test_hybrid_pipeline_retrieves_leave_policy_first(self):
        _, _, filtered = self.rag.retrieve("Can unused leave roll into next year, and is there a deadline?")
        self.assertEqual(filtered[0].chunk.document_id, "annual-leave-policy")
        self.assertTrue(filtered[0].compressed_text)

    def test_response_contains_advanced_trace(self):
        response = self.rag.ask("How quickly must a security incident be reported?", generator="local")
        steps = [item["step"] for item in response["trace"]]
        self.assertEqual(steps, ["rewrite", "hybrid_retrieve", "rerank_filter", "compress", "generate"])
        self.assertTrue(response["sources"])

    def test_vacation_synonym_returns_allowance_and_notice(self):
        response = self.rag.ask(
            "What is our vacation allowance and how far ahead must I ask?",
            generator="local",
        )
        self.assertEqual(response["sources"][0]["document_id"], "annual-leave-policy")
        self.assertIn("20 working days", response["answer"])
        self.assertIn("four weeks in advance", response["answer"])


if __name__ == "__main__":
    unittest.main()
