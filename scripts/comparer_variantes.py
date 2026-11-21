"""Compare les variantes d'entraînement VGG16 sur les mêmes images de test (2026).

Pour chaque variante : exactitude, exactitude équilibrée, rappel par classe, et test de McNemar
apparié contre la référence (R0). McNemar ne regarde que les images où les deux modèles ne sont pas
d'accord : b = R0 juste / variante fausse, c = R0 faux / variante juste.

Usage :
    python scripts/comparer_variantes.py <dossier des variantes> <dossier de sortie>
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from statsmodels.stats.contingency_tables import mcnemar

CLASSES = ["rupee_1", "rupee_2", "rupee_5", "rupee_10"]


def main(dossier: Path, sortie: Path):
    sortie.mkdir(parents=True, exist_ok=True)
    variantes = sorted(p for p in dossier.iterdir() if (p / "predictions_test.csv").exists())
    pred = {v.name: pd.read_csv(v / "predictions_test.csv").set_index("fichier") for v in variantes}
    ref_nom = variantes[0].name
    ref = pred[ref_nom]
    lignes = []
    for nom, df in pred.items():
        df = df.loc[ref.index]
        juste, juste_ref = df.vrai == df.predit, ref.vrai == ref.predit
        b, c = int((juste_ref & ~juste).sum()), int((~juste_ref & juste).sum())
        p = mcnemar([[0, b], [c, 0]], exact=True).pvalue if nom != ref_nom else None
        rappels = recall_score(df.vrai, df.predit, labels=CLASSES, average=None, zero_division=0)
        lignes.append({
            "variante": nom,
            "exactitude": round(juste.mean(), 3),
            "exactitude_equilibree": round(balanced_accuracy_score(df.vrai, df.predit), 3),
            **{f"rappel_{k}": round(r, 3) for k, r in zip(CLASSES, rappels)},
            "b_ref_juste_variante_fausse": b if p is not None else None,
            "c_ref_faux_variante_juste": c if p is not None else None,
            "mcnemar_p": round(p, 4) if p is not None else None,
        })
        pd.crosstab(df.vrai, df.predit).reindex(index=CLASSES, columns=CLASSES, fill_value=0).to_csv(
            sortie / f"confusion_{nom}.csv")
    tableau = pd.DataFrame(lignes)
    tableau.to_csv(sortie / "comparaison_variantes.csv", index=False)
    print(tableau.to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
