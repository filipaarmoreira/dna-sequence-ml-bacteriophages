import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# ── Carregar ambos os JSONs 
with open("evaluation/cv_classifiers_results.json", "r") as f:
    all_results = json.load(f)

with open("evaluation/cv_classifiers_scaled_results.json", "r") as f:
    scaled_results = json.load(f)

# Juntar tudo num dicionário único
# os scaled têm sufixo "_scaled" no nome, por isso não há conflitos
all_results.update(scaled_results)

METHODS = list(all_results.keys())

# ── Shapiro-Wilk 
non_normal = []
shapiro_results = {}

for method in METHODS:
    valores = all_results[method]["f1"]
    stat, p = stats.shapiro(valores)
    normal = p >= 0.05
    shapiro_results[method] = {"W": stat, "p": p, "normal": normal}
    flag = "normal" if normal else "NÃO normal"
    print(f"{method:55s}  W={stat:.4f}  p={p:.4f}  {flag}")
    if not normal:
        non_normal.append(method)

print(f"\nMétodos que violam normalidade: {len(non_normal)}/{len(METHODS)}")

# ── Guardar CSV com resultados 
shapiro_df = pd.DataFrame([
    {
        "method":     method,
        "W":          shapiro_results[method]["W"],
        "p_value":    shapiro_results[method]["p"],
        "normal":     shapiro_results[method]["normal"],
        "conclusion": "normal" if shapiro_results[method]["normal"] else "não normal",
        "scaled":     "_scaled" in method   # coluna extra para identificar o grupo
    }
    for method in METHODS
])
shapiro_df = shapiro_df.sort_values("p_value")
shapiro_df.to_csv("evaluation/shapiro_wilk_scaled.csv", index=False)
print("Resultados guardados em evaluation/shapiro_wilk_scaled.csv")

# ── Decisão e encaminhamento 
if non_normal:
    print("Viola normalidade — correr: python evaluation/nonparametric_tests_scaled.py")
else:
    print("Normalidade assumida — correr: python evaluation/parametric_tests_scaled.py")

# ── QQ Plots
n = len(METHODS)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
axes = axes.flatten()

for i, method in enumerate(METHODS):
    valores = all_results[method]["f1"]
    p = shapiro_results[method]["p"]
    stats.probplot(valores, dist="norm", plot=axes[i])
    color = "red" if not shapiro_results[method]["normal"] else "green"
    # indica visualmente se é scaled ou não
    label = method.replace(" : ", "\n")
    axes[i].set_title(f"{label}\nShapiro p={p:.3f}", fontsize=7, color=color)

# esconder eixos sobrantes
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("QQ Plots — Normalidade dos resíduos (F1)\n(raw vs scaled)", fontsize=12)
plt.tight_layout()
plt.savefig("evaluation/qqplot_f1_scaled.png", dpi=150, bbox_inches="tight")
plt.close()

