import os
import requests
import numpy as np
from transformers import AutoTokenizer
import pickle

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Initialiser le tokenizer
print("Loading Llama-3.2-1B-Instruct tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Encoder les données
print("Encoding data...")
train_ids = tokenizer.encode(train_data, add_special_tokens=False)
val_ids = tokenizer.encode(val_data, add_special_tokens=False)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Sauvegarder les métadonnées
meta = {
    'vocab_size': len(tokenizer),
    'tokenizer_name': "meta-llama/Llama-3.2-1B-Instruct",
    'tokenizer': tokenizer  # On sauvegarde aussi le tokenizer complet
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print(f"Saved {len(train_ids):,} training tokens")
print(f"Saved {len(val_ids):,} validation tokens")
print(f"Vocabulary size: {len(tokenizer):,} tokens")
