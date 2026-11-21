# Données

Les images ne sont pas redistribuées dans ce dépôt. Elles proviennent de jeux de données publics :

- **Brazilian coins** : 3 059 images, 5 classes (5, 10, 25, 50, 100 centavos), [kaggle.com/lgmoneda/br-coins](https://www.kaggle.com/lgmoneda/br-coins). Les images ont été rangées par classe d'après les trois premiers chiffres de leur nom (par exemple `100_1477281084` → 100 centavos).
- **Indian coins** : 656 images, 4 classes (1, 2, 5, 10 roupies), disponible sur Kaggle et IEEE DataPort.
- **Indian Coin Denomination Dataset (ICDD)** (réexécution de VGG16 en 2026) : 900 photos de 1, 2, 5, 10 et 20 roupies, rangées par pièce et par éclairage ; seules les 4 premières classes sont utilisées. Chauhan et Singh, IEEE DataPort, 2021, [doi:10.21227/t63t-5j24](http://dx.doi.org/10.21227/t63t-5j24). Préparation : `python scripts/preparer_icdd.py <dossier ICDD> <sortie>`.
- **Indian coins + fake coins** : 1 783 images, 5 classes (`1_rupee`, `2_rupee`, `5_rupee`, `invalid`, `reverse`), disponible sur Kaggle.

Les notebooks lisent les images depuis Google Drive (`/content/drive/MyDrive/...`) ; adapter ces chemins à l'emplacement des données, avec un sous-dossier par classe.

---

Images are not redistributed here. They come from the public datasets listed above; adjust the Google Drive paths in the notebooks to your local copy (one sub-folder per class).
