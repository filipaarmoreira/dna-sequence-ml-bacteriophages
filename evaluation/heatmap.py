import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp


with open("evaluation/cv_classifiers_results.json", "r") as f:
    all_results = json.load(f)

EMBEDDINGS = {
    "seq2vec":                "Seq2Vec",
    "hyenadna":               "HyenaDNA",
    "nucleotide_transformer": "Nucleotide Transformer",
    "dnabert2_mean":          "DNABERT-2 (mean pooling)",
    
}


rows = []
for key, vals in all_results.items():
    for i, score in enumerate(vals["f1"]):
        rows.append({"method": key, "cv_cycle": i, "f1": score})

df_all = pd.DataFrame(rows)


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def make_heatmap(df, title, output_path):
    means = df.groupby("method")["f1"].mean().sort_values(ascending=False)
    order = list(means.index)

    # Conover-Friedman (não paramétrico) 
    # Usado para todos os métodos pois alguns violam normalidade
    pivot = df.pivot(index="cv_cycle", columns="method", values="f1")[order]
    pc = sp.posthoc_conover_friedman(pivot, p_adjust="holm")

    effect_matrix = pd.DataFrame(0.0, index=order, columns=order)
    pval_matrix   = pd.DataFrame(1.0, index=order, columns=order)

    for i, m1 in enumerate(order):
        for j, m2 in enumerate(order):
            if i < j:
                diff = means[m1] - means[m2]
                p    = pc.loc[m1, m2]
                effect_matrix.loc[m1, m2] =  diff
                effect_matrix.loc[m2, m1] = -diff
                pval_matrix.loc[m1, m2]   =  p
                pval_matrix.loc[m2, m1]   =  p

    annot = pd.DataFrame("", index=order, columns=order)
    for m1 in order:
        for m2 in order:
            if m1 == m2:
                annot.loc[m1, m2] = f"{means[m1]:.3f}"
            else:
                d = effect_matrix.loc[m1, m2]
                p = pval_matrix.loc[m1, m2]
                annot.loc[m1, m2] = f"{d:.2f}\n{sig(p)}"

    # Labels simplificados
    tick_labels = []
    for m in order:
        clf = m.split(" : ")[1] if " : " in m else m
        tick_labels.append(f"{clf}\n(F1={means[m]:.3f})")

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        effect_matrix,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        center=0,
        vmin=-0.3, vmax=0.3,
        linewidths=0.5,
        linecolor="white",
        square=True,
        ax=ax,
        cbar_kws={"label": "Effect size (mean F1 diff)", "shrink": 0.8},
        annot_kws={"size": 10},
    )

    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Nota de rodapé
    fig.text(0.01, 0.01, "Statistical significance: Conover-Friedman + Holm-Bonferroni correction",
             fontsize=7, color="grey")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")

# Gerar um heatmap por embedding 
for enc_key, enc_label in EMBEDDINGS.items():
    df_enc = df_all[df_all["method"].str.startswith(enc_key)].copy()

    if df_enc.empty:
        print(f"Sem dados para {enc_key}")
        continue

    title = f"MCSim Plot — {enc_label}\nEffect size (color) + Statistical significance (stars)"
    output = f"evaluation/mcsim_{enc_key}_final.png"
    make_heatmap(df_enc, title, output)