"""Exporte le VGG16 des pièces au format ONNX pour la démo dans le navigateur (2026).

Le modèle est reconstruit (VGG16 gelé + Flatten + Dense(4)), ses poids sont rechargés, puis il est
converti en ONNX. Une version quantifiée (poids en entiers sur 8 bits) divise la taille par environ 4,
pour un téléchargement raisonnable dans le navigateur.

Entrée du modèle ONNX : image RGB 224 x 224, valeurs divisées par 255, format NHWC (1, 224, 224, 3).
Sortie : probabilités des classes, dans l'ordre alphabétique des dossiers (rupee_1, rupee_10, rupee_2, rupee_5).

Usage :
    python scripts/exporter_onnx.py <vgg16.weights.h5> <dossier de sortie>
"""
import sys
from pathlib import Path

import tensorflow as tf
import tf2onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model


def main(poids: Path, sortie: Path):
    sortie.mkdir(parents=True, exist_ok=True)
    base = VGG16(input_shape=(224, 224, 3), weights=None, include_top=False)
    modele = Model(base.input, Dense(4, activation="softmax")(Flatten()(base.output)))
    modele.load_weights(poids)

    fonction = tf.function(lambda x: modele(x, training=False),
                           input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name="image")])
    complet = sortie / "pieces_vgg16.onnx"
    tf2onnx.convert.from_function(fonction, input_signature=fonction.input_signature, opset=17,
                                  output_path=str(complet))
    quantifie = sortie / "pieces_vgg16_int8.onnx"
    quantize_dynamic(str(complet), str(quantifie), weight_type=QuantType.QUInt8)
    for f in (complet, quantifie):
        print(f"{f.name} : {f.stat().st_size / 1e6:.1f} Mo")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
