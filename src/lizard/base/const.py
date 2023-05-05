import src.lizard.llm as llm

models = {
    "gpt2": {
        "tokenizer": llm.GPT2Tokenizer,
        "model": llm.GPT2LMHeadModel,
        "config": llm.GPT2Config,
    },
    "gpt2-medium": {
        "tokenizer": llm.GPT2Tokenizer,
        "model": llm.GPT2LMHeadModel,
        "config": llm.GPT2Config,
    },
    "gpt2-large": {
        "tokenizer": llm.GPT2Tokenizer,
        "model": llm.GPT2LMHeadModel,
        "config": llm.GPT2Config,
    },
    "gpt2-xl": {
        "tokenizer": llm.GPT2Tokenizer,
        "model": llm.GPT2LMHeadModel,
        "config": llm.GPT2Config,
    },
}