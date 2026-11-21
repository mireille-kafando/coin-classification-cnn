import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage


def recadrer(image, taille=256, marge=1.08):
    """Recadre la photo sur la pièce, puis noircit l'extérieur du disque.

    La pièce est la zone riche en contours (relief, inscriptions) ; la feuille de fond est lisse. On garde la plus
    grande région de contours ; si elle couvre presque toute l'image (gros plan), on prend le carré central.
    """
    rgb = image.convert("RGB")
    gris = np.asarray(rgb.convert("L").filter(ImageFilter.GaussianBlur(2)), dtype=float)
    gx, gy = ndimage.sobel(gris, 1), ndimage.sobel(gris, 0)
    force = np.hypot(gx, gy)
    masque = force > max(40.0, np.percentile(force, 90))
    masque = ndimage.binary_closing(masque, iterations=6)
    masque = ndimage.binary_fill_holes(masque)
    etiquettes, n = ndimage.label(masque)
    h, w = gris.shape
    if n:
        tailles = ndimage.sum(masque, etiquettes, range(1, n + 1))
        ys, xs = ndimage.find_objects((etiquettes == int(np.argmax(tailles)) + 1).astype(int))[0]
        cy, cx = (ys.start + ys.stop) / 2, (xs.start + xs.stop) / 2
        demi = max(ys.stop - ys.start, xs.stop - xs.start) / 2 * marge
    if not n or demi * 2 > 0.8 * min(h, w):  # gros plan : la pièce occupe déjà l'image
        cy, cx, demi = h / 2, w / 2, min(h, w) / 2
    carre = rgb.crop((int(cx - demi), int(cy - demi), int(cx + demi), int(cy + demi))).resize((taille, taille), Image.LANCZOS)
    disque = Image.new("L", (taille, taille), 0)
    ImageDraw.Draw(disque).ellipse([0, 0, taille - 1, taille - 1], fill=255)
    fond = Image.new("RGB", (taille, taille), (0, 0, 0))
    fond.paste(carre, (0, 0), disque)
    return fond
