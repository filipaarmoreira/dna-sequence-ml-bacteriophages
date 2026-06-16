import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def seq_to_kmers(sequence, k=4):
    # converte uma sequência DNA numa string de k-mers separados por espaço
    sequence = sequence.upper()
    return " ".join(sequence[i:i+k] for i in range(len(sequence) - k + 1))


df = pd.read_csv("dataset/processed/final_dataset.csv")
sequences = df["Sequence"].tolist()

# converter todas as sequências em k-mers
kmer_sequences = [seq_to_kmers(seq, k=4) for seq in sequences]

# CountVectorizer conta a frequência de cada k-mer em cada sequência
# fit_transform aprende o vocabulário (todos os k-mers únicos) e conta as frequências
# resultado: matriz (n_sequências, n_kmers_únicos)
vectorizer = CountVectorizer(analyzer="word", ngram_range=(1,1))
emb_matrix = vectorizer.fit_transform(kmer_sequences).toarray()



result = pd.concat([
    df[["Accession_number", "Label"]].reset_index(drop=True),
    pd.DataFrame(emb_matrix)
], axis=1)


result.to_csv("embeddings/embeddings_seq2vec.csv", index=False)
