import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_core import NaiveRAG


class NaiveRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = NaiveRAG(ROOT / "documents", top_k=3)

    def test_loads_synthetic_corpus(self):
        self.assertEqual(len(self.rag.documents), 8)
        self.assertGreaterEqual(len(self.rag.chunks), 24)

    def test_leave_query_retrieves_leave_policy(self):
        results = self.rag.index.search("carry over unused annual leave deadline", top_k=3)
        self.assertEqual(results[0].chunk.document_id, "annual-leave-policy")

    def test_offline_answer_contains_sources_and_trace(self):
        response = self.rag.ask("How many annual leave days may be carried over?", generator="local")
        self.assertTrue(response["answer"])
        self.assertEqual(len(response["sources"]), 3)
        self.assertEqual(response["trace"][0]["step"], "retrieve")


if __name__ == "__main__":
    unittest.main()
