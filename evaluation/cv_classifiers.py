import pandas as pd
import numpy as np
import json
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

ENCODERS = {
    "seq2vec":                "embeddings/embeddings_seq2vec.csv",
    "dnabert2_mean":          "embeddings/embeddings_dnabert2_mean.csv",
    "dnabert2_max":           "embeddings/embeddings_dnabert2_max.csv",
    "hyenadna":               "embeddings/embeddings_hyenadna.csv",
    "nucleotide_transformer": "embeddings/embeddings_nt.csv"
}

# todos têm random_state=42 para garantir que os resultados são reprodutíveis
# e que as diferenças entre encoders não são devidas à aleatoriedade do treino
CLASSIFIERS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost":       XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
    "svm":           SVC(random_state=42),
    "mlp":           MLPClassifier(random_state=42, max_iter=1000),
}

all_results = {}

for enc_name, path in ENCODERS.items():
    df       = pd.read_csv(path)
    features = df.drop(columns=["Accession_number", "Label"]).values
    labels   = (df["Label"] == "Lytic").astype(int).values  # lytic=1, lysogenic=0

    for clf_name, clf in CLASSIFIERS.items():
        name      = f"{enc_name} : {clf_name}"
        f1_scores = []

        for train_idx, test_idx in cv.split(features, labels):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx],   labels[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, average="macro"))

        all_results[name] = {"f1": f1_scores}
        print(f"{name}")
        print(f"  F1 macro: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

with open("evaluation/cv_classifiers_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

