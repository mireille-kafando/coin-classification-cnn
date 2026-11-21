"""Prépare le jeu ICDD pour la réexécution de VGG16 (2026) : 4 classes, partage par pièce physique.

Le jeu d'origine du notebook 02 (656 images, 4 classes) n'a pas été conservé. La réexécution utilise
le jeu public Indian Coin Denomination Dataset (ICDD, Chauhan et Singh, IEEE DataPort, 2021,
doi:10.21227/t63t-5j24) : photos de pièces rangées par dénomination, par pièce (« Coin k ») et par
éclairage. Seules les classes du projet sont gardées (1, 2, 5 et 10 roupies ; les 20 roupies sont
écartées) et renommées comme dans le notebook (rupee_1, rupee_2, rupee_5, rupee_10).

Partage par pièce physique : toutes les photos d'une même pièce vont dans le même ensemble. Pour chaque
classe, une pièce tirée au hasard (graine 2026) va au test, une autre à la validation, les autres à
l'entraînement. Ainsi, le test mesure la reconnaissance de pièces jamais vues, et non de photos
presque identiques à celles de l'entraînement.

Usage :
    python scripts/preparer_icdd.py <dossier ICDD réduit> <dossier de sortie>
<dossier ICDD réduit> contient <Classe>/<Coin_k>/<Éclairage>/<image>.jpg (One, Two, Five, Ten).
Écrit <sortie>/{train,val,test}/rupee_*/ et <sortie>/partage_icdd.json.
"""
import json
import random
import shutil
import sys
from pathlib import Path

CLASSES = {"One": "rupee_1", "Two": "rupee_2", "Five": "rupee_5", "Ten": "rupee_10"}
GRAINE = 2026


def main(source: Path, sortie: Path):
    rng = random.Random(GRAINE)
    partage = {}
    for dossier, classe in CLASSES.items():
        pieces = sorted(p.name for p in (source / dossier).iterdir() if p.is_dir())
        tirage = pieces[:]
        rng.shuffle(tirage)
        affectation = {"test": [tirage[0]], "val": [tirage[1]], "train": sorted(tirage[2:])}
        partage[classe] = affectation
        for ensemble, liste in affectation.items():
            cible = sortie / ensemble / classe
            cible.mkdir(parents=True, exist_ok=True)
            for piece in liste:
                for image in sorted((source / dossier / piece).rglob("*.jpg")):
                    eclairage = image.parent.name
                    shutil.copy(image, cible / f"{piece}_{eclairage}_{image.name}")
    comptes = {e: {c: len(list((sortie / e / c).glob("*.jpg"))) for c in CLASSES.values()}
               for e in ("train", "val", "test")}
    (sortie / "partage_icdd.json").write_text(json.dumps(
        {"graine": GRAINE, "pieces": partage, "images": comptes}, indent=1, ensure_ascii=False))
    print(json.dumps(comptes, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
