import unittest
from src.lizard.llm import LLM


class TestLLM(unittest.TestCase):
    def test_initialization(self):
        llm = LLM("gpt2")
        self.assertIsInstance(llm, LLM)

    def test_fit(self):
        llm = LLM("gpt2", epochs=1)
        texts = ["This is a test text."] * 10
        llm.fit(texts)
        self.assertTrue(llm.model is not None)

    def test_transform(self):
        llm = LLM("gpt2", epochs=1)
        texts = ["This is a test text."] * 10
        llm.fit(texts)
        generated_texts = llm.transform(["Test input"])
        self.assertIsInstance(generated_texts, list)
        self.assertEqual(len(generated_texts), 1)
        self.assertIsInstance(generated_texts[0], str)



if __name__ == "__main__":
    unittest.main()
