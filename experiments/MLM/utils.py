from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir='.cache')
encoded_input = tokenizer('hello [SEP] hi')
print(encoded_input)
