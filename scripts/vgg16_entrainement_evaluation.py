"""VGG16 par transfert sur les pièces indiennes, avec une évaluation corrigée (2026).

Reprend le protocole du notebook 02 (prétraitement, augmentation, VGG16 gelé + Flatten + Dense(4),
Adam 3e-4, 20 époques, 20 lots d'entraînement et 10 de validation par époque), avec trois corrections :

1. Le générateur de test est créé avec shuffle=False. Dans le notebook, il mélangeait les images
   (valeur par défaut), si bien que les prédictions de model.predict(test_set) ne correspondaient plus
   aux étiquettes test_set.classes, rangées dans l'ordre des dossiers. L'exactitude de test de 20,6 %
   et la matrice de confusion du notebook ne mesurent donc pas le modèle.
2. La métrique F1 vient de Keras (keras.metrics.F1Score) et non de tensorflow-addons, abandonné.
3. La deuxième figure est légendée « rappel » (elle l'était « prec ») et les poids sont sauvegardés.

Usage :
    python scripts/vgg16_entrainement_evaluation.py <dossier> [--sortie resultats_vgg16]
où <dossier> contient train/, val/ et test/, avec un sous-dossier par classe (532, 152 et 68 images
dans le projet). Les images ne sont pas redistribuées dans ce dépôt (voir data/README.md).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = [224, 224]


def resize_and_rescale(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    return image / 255.0


def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0] + 6, IMG_SIZE[1] + 6)
    image = tf.image.random_crop(image, size=[IMG_SIZE[0], IMG_SIZE[1], 3])
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.adjust_hue(image, 0.04)
    return resize_and_rescale(image)


def generateurs(dossier, graine):
    def flux(sous_dossier, fonction, melange):
        return ImageDataGenerator(preprocessing_function=fonction).flow_from_directory(
            str(dossier / sous_dossier), target_size=tuple(IMG_SIZE), batch_size=10,
            class_mode="categorical", shuffle=melange, seed=graine)

    # Correction 1 : le test n'est pas mélangé, pour que les prédictions suivent test_set.classes
    return flux("train", augment, True), flux("val", resize_and_rescale, True), \
        flux("test", resize_and_rescale, False)


def construire(n_classes, poids):
    base = VGG16(input_shape=IMG_SIZE + [3], weights=poids, include_top=False)
    base.trainable = False
    sortie = Dense(n_classes, activation="softmax")(Flatten()(base.output))
    modele = Model(inputs=base.input, outputs=sortie)
    modele.compile(keras.optimizers.Adam(learning_rate=0.0003),
                   loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                   metrics=[keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"),
                            keras.metrics.F1Score(name="f1_score"),  # correction 2
                            keras.metrics.CategoricalAccuracy(name="categorical_accuracy")])
    return modele


def tracer(historique, cle, titre, fichier):
    plt.figure()
    plt.plot(historique[cle], label=f"{titre} (entraînement)")
    plt.plot(historique[f"val_{cle}"], label=f"{titre} (validation)")
    plt.xlabel("Époque")
    plt.legend()
    plt.savefig(fichier, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dossier", type=Path)
    p.add_argument("--sortie", type=Path, default=Path("resultats_vgg16"))
    p.add_argument("--epoques", type=int, default=20)
    p.add_argument("--graine", type=int, default=2000)
    p.add_argument("--poids", default="imagenet", help="« imagenet » (par défaut) ou « aucun » pour un test rapide")
    a = p.parse_args()

    keras.utils.set_random_seed(a.graine)
    a.sortie.mkdir(parents=True, exist_ok=True)
    entrainement, validation, test = generateurs(a.dossier, a.graine)
    modele = construire(entrainement.num_classes, None if a.poids == "aucun" else a.poids)
    historique = modele.fit(entrainement, validation_data=validation, epochs=a.epoques,
                            steps_per_epoch=min(20, len(entrainement)),
                            validation_steps=min(10, len(validation))).history

    tracer(historique, "precision", "Précision", a.sortie / "vgg16_precision.png")
    tracer(historique, "recall", "Rappel", a.sortie / "vgg16_rappel.png")  # correction 3
    modele.save_weights(a.sortie / "vgg16.weights.h5")

    probabilites = modele.predict(test)
    y_pred = probabilites.argmax(axis=1)
    y_vrai = test.classes  # même ordre que les prédictions grâce à shuffle=False
    noms = list(test.class_indices)
    ConfusionMatrixDisplay(confusion_matrix(y_vrai, y_pred), display_labels=noms).plot(values_format="d")
    plt.savefig(a.sortie / "vgg16_matrice_confusion_test.png", dpi=120, bbox_inches="tight")
    resultats = {
        "exactitude_test": float(accuracy_score(y_vrai, y_pred)),
        "exactitude_equilibree_test": float(balanced_accuracy_score(y_vrai, y_pred)),
        "rapport": classification_report(y_vrai, y_pred, target_names=noms, output_dict=True, zero_division=0),
        "evaluation_validation": dict(zip(modele.metrics_names, map(lambda v: np.asarray(v).tolist(),
                                                                    modele.evaluate(validation, verbose=0)))),
    }
    (a.sortie / "resultats_test.json").write_text(json.dumps(resultats, indent=1, ensure_ascii=False))
    print(classification_report(y_vrai, y_pred, target_names=noms, zero_division=0))
    print(f"Exactitude sur le test : {resultats['exactitude_test']:.3f}")


if __name__ == "__main__":
    main()
