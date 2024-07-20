import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

def read_rulebooks(directory):
    texts = {}
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                texts[file] = f.read()
    return texts

def get_embedding(text):
    text = text.replace("\n", " ")
    if not text.strip():
        return None
    return embed_model.encode(text, normalize_embeddings=True)

def save_embeddings(directory):
    rulebook_texts = read_rulebooks(directory)
    embeddings = {}

    for file, text in tqdm(rulebook_texts.items(), desc="Generating Embeddings", unit="file"):
        embeddings[file] = get_embedding(text)
    
    with open('rulebook_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)

if __name__ == "__main__":
    save_embeddings('books')
