import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42) #42 mas podia ser outro qualquer mas ao colocar isto faz com que o valor seja sempre o mesmo de cada vez que corro

# Top 4 modelos sem standardização
BEST_MODELS = {
    "Seq2Vec + XGBoost":       ("embeddings/embeddings_seq2vec.csv",  XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)),
    "Seq2Vec + Random Forest":  ("embeddings/embeddings_seq2vec.csv",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    "HyenaDNA + XGBoost":      ("embeddings/embeddings_hyenadna.csv", XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)),
    "HyenaDNA + Random Forest": ("embeddings/embeddings_hyenadna.csv", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
}

CLASS_NAMES = ["Lysogenic", "Lytic"]  # 0 = Lysogenic, 1 = Lytic

for model_name, (path, clf) in BEST_MODELS.items():

    df       = pd.read_csv(path)
    features = df.drop(columns=["Accession_number", "Label"]).values
    labels   = (df["Label"] == "Lytic").astype(int).values #litico 1 e lisogenico 0

    # Agregar predições de todos os folds — mais representativo que um fold único
    all_true = []
    all_pred = []

    for train_idx, test_idx in cv.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx],   labels[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    # Calcular matriz de confusão agregada
    cm      = confusion_matrix(all_true, all_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) # normalizar por linha (proporção por classe real)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm_norm,
        annot=False,
        fmt="",
        cmap="Blues",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="white",
        square=True,
        ax=ax,
        cbar_kws={"label": "Proportion", "shrink": 0.8},
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center",
                    fontsize=11, color="black")

    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11, rotation=0)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()

    fname  = model_name.lower().replace(" + ", "_").replace(" ", "_")
    output = f"evaluation/confusion_matrix_{fname}.png"
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
