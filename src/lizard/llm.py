import time
import io
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from transformers import TextGenerationPipeline, TrainingArguments, Trainer

from base.datasets import TextDataset
from base.const import models


class LLM(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, epochs=3, learning_rate=2e-5):
        self.model_name = model_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tokenizer = models[model_name]["tokenizer"].from_pretrained(model_name)
        self.config = models[model_name]["config"].from_pretrained(model_name)
        self.model = models[model_name]["model"].from_pretrained(model_name, config=self.config)
        self.inmemory_file = io.StringIO()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def fit(self, X, y=None):
        def tokenize_function(examples):
            return self.tokenizer(examples, truncation=True, padding="max_length", max_length=self.config.n_positions)

        X_tokenized = tokenize_function(X)

        training_args = TrainingArguments(
            output_dir="./gpt2_finetuned",
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=8,
            logging_dir="./logs",
            logging_steps=100,
            save_steps=0,
            do_train=True,
            do_eval=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=TextDataset(X_tokenized, y),
        )

        trainer.train()
        return self

    def transform(self, X):
        text_gen_pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)
        generated_texts = text_gen_pipeline(X, max_length=self.config.n_positions)
        result = np.array([text[0]["generated_text"] for text in generated_texts])
        return result


    def transform_one_by_one(self, input_text):
        def generate_token(model, input_ids, past=None):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, past_key_values=past)
            return outputs

        self.model.eval()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        past_key_values = None
        max_length = self.config.n_positions

        for _ in range(max_length):
            outputs = generate_token(self.model, input_ids, past=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze()
            input_ids = token.unsqueeze(0).unsqueeze(0)

            # Update the internal result variable
            self.inmemory_file.write(self.tokenizer.decode([token.item()], clean_up_tokenization_spaces=False))

            if token.item() == self.tokenizer.eos_token_id:
                break

    def poll_result(self, interval=0.5):
        prev_position = 0
        while True:
            self.inmemory_file.seek(prev_position)
            new_data = self.inmemory_file.read()
            if new_data:
                print(new_data, end="")
                prev_position = self.inmemory_file.tell()
            time.sleep(interval)