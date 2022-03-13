from pathlib import Path
from transformers import BertTokenizer, BertModel
import yaml
import pickle

out_file = 'naics_encoded.json'
encoded_naics = {}

with open('Naics.yaml', 'r') as f:
    naics = yaml.load(f, Loader=yaml.Loader)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

for k, v in naics.items():
    for k2, desc in v.items():
        if k2 != 'desc':
            encoded_input = tokenizer(desc, return_tensors='pt')
            output = list(model(**encoded_input).pooler_output[0, ...].detach().numpy())
            encoded_naics[k2] = output

with open(out_file, 'wb') as f:
    pickle.dump(encoded_naics, f)
