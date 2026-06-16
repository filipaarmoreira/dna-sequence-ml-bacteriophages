import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# triton desinstalado para evitar incompatibilidade com flash attention 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)
model.eval()

df = pd.read_csv("dataset/processed/final_dataset.csv")
sequences = df["Sequence"].tolist()

all_embeddings_mean = []
all_embeddings_max = []

for i, dna in enumerate(sequences):
    inputs = tokenizer(dna, return_tensors='pt', truncation=True, max_length=4000)["input_ids"].to(device)
    ##tensores pytorch, truncation permite cortar as seq que são menores que o max_length, inputs_ids só ids dos tokens
    hidden_states = model(inputs)[0]  

    # embedding with mean pooling
    embedding_mean = torch.mean(hidden_states[0], dim=0) # primeiro remove batch com [0], depois dim=0 são os tokens
    # é o mesmo que hidden_states = self.model(input_ids)[0] embedding = torch.max(hidden_states, dim=1)[0].squeeze().cpu().numpy()
    # embedding with max pooling
    embedding_max = torch.max(hidden_states[0], dim=0)[0]

    all_embeddings_mean.append(embedding_mean.detach().cpu().numpy())
    all_embeddings_max.append(embedding_max.detach().cpu().numpy()) #detach desliga p tensor, move para cpu e array para csv

    

emb_matrix_mean = np.vstack(all_embeddings_mean)
result_mean = pd.concat([
    df[["Accession_number", "Label"]].reset_index(drop=True),
    pd.DataFrame(emb_matrix_mean)], axis=1)
result_mean.to_csv("embeddings/embeddings_dnabert2_mean.csv", index=False)


emb_matrix_max = np.vstack(all_embeddings_max)
result_max = pd.concat([
    df[["Accession_number", "Label"]].reset_index(drop=True),
    pd.DataFrame(emb_matrix_max)], axis=1)
result_max.to_csv("embeddings/embeddings_dnabert2_max.csv", index=False)

