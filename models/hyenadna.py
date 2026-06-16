import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True)
model = AutoModel.from_pretrained("LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True)
model.eval()

df = pd.read_csv("dataset/processed/final_dataset.csv")
sequences = df["Sequence"].tolist()

all_embeddings = []
for i, dna in enumerate(sequences):
    inputs = tokenizer(dna, return_tensors="pt")["input_ids"] #como aceita até 1milhão não preciso de colocar truncation nem max length
    with torch.inference_mode():
        embeddings = model(inputs)
    all_embeddings.append(embeddings.last_hidden_state[0].mean(dim=0).numpy())
    
    
emb_matrix = np.vstack(all_embeddings)
result = pd.concat([
    df[["Accession_number", "Label"]].reset_index(drop=True),
    pd.DataFrame(emb_matrix)
], axis=1)
result.to_csv("embeddings/embeddings_hyenadna.csv", index=False)
