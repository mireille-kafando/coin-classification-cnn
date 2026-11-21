"""Résume la validation croisée par pièce (2026) : 4 plis, chaque pli teste une autre pièce par classe.

Pour chaque variante : exactitude et rappel des 2 roupies par pli, puis sur les 480 images de test
réunies (chaque image n'est testée qu'une fois). Test de McNemar exact apparié contre la référence,
sur ces 480 images.

Usage :
    python scripts/validation_croisee.py <sortie> <nom>=<pli0>,<pli1>,<pli2>,<pli3> [...]
où chaque <pli> est un dossier contenant predictions_test.csv. La première variante sert de référence.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from statsmodels.stats.contingency_tables import mcnemar

CLASSES = ["rupee_1", "rupee_2", "rupee_5", "rupee_10"]


def lire(dossiers):
    morceaux = []
    for k, d in enumerate(dossiers):
        df = pd.read_csv(Path(d) / "predictions_test.csv")
        df["pli"] = k
        df["cle"] = df.pli.astype(str) + "/" + df.fichier
        morceaux.append(df)
    return pd.concat(morceaux).set_index("cle")


def main(sortie: Path, variantes):
    sortie.mkdir(parents=True, exist_ok=True)
    donnees = {nom: lire(dossiers) for nom, dossiers in variantes}
    ref_nom, ref = variantes[0][0], donnees[variantes[0][0]]
    plis, resume = [], []
    for nom, df in donnees.items():
        df = df.loc[ref.index]
        for k, g in df.groupby("pli"):
            plis.append({"variante": nom, "pli": k, "exactitude": round((g.vrai == g.predit).mean(), 3),
                         "rappel_rupee_2": round(recall_score(g.vrai, g.predit, labels=["rupee_2"], average=None,
                                                              zero_division=0)[0], 3)})
        juste, juste_ref = df.vrai == df.predit, ref.vrai == ref.predit
        b, c = int((juste_ref & ~juste).sum()), int((~juste_ref & juste).sum())
        exactitudes = [p["exactitude"] for p in plis if p["variante"] == nom]
        rappels = recall_score(df.vrai, df.predit, labels=CLASSES, average=None, zero_division=0)
        resume.append({
            "variante": nom, "images": len(df),
            "exactitude": round(juste.mean(), 3),
            "exactitude_min_pli": min(exactitudes), "exactitude_max_pli": max(exactitudes),
            "exactitude_equilibree": round(balanced_accuracy_score(df.vrai, df.predit), 3),
            **{f"rappel_{k}": round(r, 3) for k, r in zip(CLASSES, rappels)},
            "b_ref_juste_variante_fausse": b if nom != ref_nom else None,
            "c_ref_faux_variante_juste": c if nom != ref_nom else None,
            "mcnemar_p": round(mcnemar([[0, b], [c, 0]], exact=True).pvalue, 4) if nom != ref_nom else None,
        })
        pd.crosstab(df.vrai, df.predit).reindex(index=CLASSES, columns=CLASSES, fill_value=0).to_csv(
            sortie / f"confusion_cv_{nom}.csv")
    pd.DataFrame(plis).to_csv(sortie / "validation_croisee_par_pli.csv", index=False)
    pd.DataFrame(resume).to_csv(sortie / "validation_croisee_resume.csv", index=False)
    print(pd.DataFrame(plis).pivot(index="pli", columns="variante").to_string())
    print(pd.DataFrame(resume).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [(a.split("=")[0], a.split("=")[1].split(",")) for a in sys.argv[2:]])
