import json
import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp


with open("evaluation/cv_classifiers_results.json", "r") as f:
    all_results = json.load(f)

METHODS = list(all_results.keys())


rows = []
for method in METHODS:
    for i, score in enumerate(all_results[method]["f1"]):
        rows.append({"method": method, "cv_cycle": i, "f1": score})

df = pd.DataFrame(rows)
means = df.groupby("method")["f1"].mean().sort_values(ascending=False)
order = list(means.index)

# Friedman test 
# Como alguns métodos violam normalidade, aplicamos testes não paramétricos a todos


friedman = pg.friedman(df, dv="f1", within="method", subject="cv_cycle")
p_friedman = friedman["p_unc"].values[0]
chi2 = friedman["Q"].values[0]
print(f"  Chi2 = {chi2:.4f}")
print(f"  p    = {p_friedman:.4e}")

if p_friedman >= 0.05:
    print("\n  Não significativo — sem diferenças entre métodos.")
    exit()

# Conover-Friedman com correcção de Holm 

print("\nPOST-HOC: Conover-Friedman (Holm correction)")
pivot = df.pivot(index="cv_cycle", columns="method", values="f1")[order]
pc = sp.posthoc_conover_friedman(pivot, p_adjust="holm")


rows_out = []
found = False
for i, m1 in enumerate(order):
    for j, m2 in enumerate(order):
        if i < j:
            p = pc.loc[m1, m2]
            diff = means[m1] - means[m2]
            rows_out.append({
                "A": m1, "B": m2,
                "p_corr": round(p, 4),
                "delta_f1": round(diff, 4)
            })
            if p < 0.05:
                print(f"  {m1} vs {m2}: p={p:.4f}  F1={diff:+.4f}")
                found = True

if not found:
    print("  Nenhum par significativo encontrado.")


results_df = pd.DataFrame(rows_out)
results_df.to_csv("evaluation/conover_f1_final.csv", index=False)
print("\nResultados guardados em evaluation/conover_f1_final.csv")


print("\nRESUMO POR MÉTODO (ordenado por F1 médio):")
for i, method in enumerate(order):
    vals = all_results[method]["f1"]
    print(f"  {i+1:2d}. {method:45s}  {means[method]:.4f} ± {np.std(vals):.4f}")