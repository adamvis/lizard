import unittest
from transformers import AutoTokenizer
from src.lizard.base.datasets import TextDataset


class TestTextDataset(unittest.TestCase):
    def test_initialization(self):
        tokenized_texts = AutoTokenizer.from_pretrained("gpt2")(["This is a test text."] * 5)
        text_dataset = TextDataset(tokenized_texts)
        self.assertIsInstance(text_dataset, TextDataset)

    def test_len(self):
        tokenized_texts = AutoTokenizer.from_pretrained("gpt2")(["This is a test text."] * 5)
        text_dataset = TextDataset(tokenized_texts)
        self.assertEqual(len(text_dataset), 5)

    def test_getitem(self):
        tokenized_texts = AutoTokenizer.from_pretrained("gpt2")(["This is a test text."] * 5)
        text_dataset = TextDataset(tokenized_texts)
        item = text_dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)


if __name__ == "__main__":
    unittest.main()
