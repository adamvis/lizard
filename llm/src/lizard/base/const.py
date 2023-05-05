import transformers

models = {
    "gpt2": {
        "tokenizer": transformers.GPT2Tokenizer,
        "model": transformers.GPT2LMHeadModel,
        "config": transformers.GPT2Config,
    },
    "gpt2-medium": {
        "tokenizer": transformers.GPT2Tokenizer,
        "model": transformers.GPT2LMHeadModel,
        "config": transformers.GPT2Config,
    },
    "gpt2-large": {
        "tokenizer": transformers.GPT2Tokenizer,
        "model": transformers.GPT2LMHeadModel,
        "config": transformers.GPT2Config,
    },
    "gpt2-xl": {
        "tokenizer": transformers.GPT2Tokenizer,
        "model": transformers.GPT2LMHeadModel,
        "config": transformers.GPT2Config,
    },
}