from tokenizers import Tokenizer
from transformers import AutoTokenizer, BitsAndBytesConfig

model_path = "mtgv/MobileVLM-1.7B"
input = ['[12,3.2,56,44].', '[15,,56,44].']
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
input_id = tokenizer(input)
print(input_id)


