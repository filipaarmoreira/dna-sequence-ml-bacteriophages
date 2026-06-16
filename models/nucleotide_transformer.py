import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True).to(device)
model.eval()

max_length = 4096 

df = pd.read_csv("dataset/processed/final_dataset.csv")
sequences = df["Sequence"].tolist()

all_embeddings = []

for i, dna in enumerate(sequences):
    tokens_ids = tokenizer.batch_encode_plus(
        [dna],
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    )["input_ids"].to(device)

    attention_mask = (tokens_ids != tokenizer.pad_token_id).to(device)

    with torch.no_grad():
        torch_outs = model(
            tokens_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )

    embeddings = torch_outs['hidden_states'][-1]
    attention_mask_exp = torch.unsqueeze(attention_mask, dim=-1)
    mean_embedding = torch.sum(attention_mask_exp * embeddings, axis=-2) / torch.sum(attention_mask, axis=1, keepdim=True)
    all_embeddings.append(mean_embedding.squeeze().cpu().numpy())

    if i % 50 == 0:
        print(f"  {i}/{len(sequences)}")

emb_matrix = np.vstack(all_embeddings)
result = pd.concat([
    df[["Accession_number", "Label"]].reset_index(drop=True),
    pd.DataFrame(emb_matrix)
], axis=1)

result.to_csv("embeddings/embeddings_nt.csv", index=False)
