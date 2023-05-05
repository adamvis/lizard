import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
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

        if type(generated_texts) is not str:
            result = [text["generated_text"] for text in generated_texts]
        else:
            result = generated_texts
            
        return result
