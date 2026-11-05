"""
07 - Verificación de la cuantización de pesos
=============================================
Compara la exactitud del modelo original (coma flotante de 32 bits) frente a
sus versiones comprimidas para el navegador:
  - f16 : coma flotante de 16 bits (peso ~2 veces menor)
  - i8  : enteros de 8 bits con escala por tensor (peso ~4 veces menor)

Se usa el conjunto de prueba completo. Sirve para elegir el formato que se
incrusta en el archivo HTML final.

Uso:
    python entrenamiento/07_verificar_cuantizacion.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE = Path(__file__).resolve().parent.parent
DIR_DATOS = BASE / "datos" / "procesado"
DIR_RESULTADOS = BASE / "entrenamiento" / "resultados"
IMG = 224
MEDIA = [0.485, 0.456, 0.406]
DESV = [0.229, 0.224, 0.225]

torch.set_num_threads(16)


def es_kernel(nombre, forma) -> bool:
    """Son 'kernels' los pesos de convoluciones (4D) y capas densas (2D)."""
    return nombre.endswith(".weight") and len(forma) >= 2


def cuantizar(estado, formato):
    """Devuelve una copia del state_dict con los tensores cuantizados."""
    nuevo = {}
    for nombre, tensor in estado.items():
        arr = tensor.numpy()
        if formato == "i8" and es_kernel(nombre, arr.shape):
            escala = float(np.max(np.abs(arr)) / 127.0) or 1.0
            arr = np.clip(np.round(arr / escala), -127, 127) * escala
        elif formato == "f16" and es_kernel(nombre, arr.shape):
            arr = arr.astype(np.float16).astype(np.float32)
        nuevo[nombre] = torch.from_numpy(np.ascontiguousarray(arr))
    return nuevo


@torch.no_grad()
def evaluar(modelo, cargador):
    modelo.eval()
    correctos = top5_correctos = total = 0
    for x, y in cargador:
        logits = modelo(x)
        pred = logits.topk(5, dim=1).indices
        correctos += (pred[:, 0] == y).sum().item()
        top5_correctos += (pred == y[:, None]).any(dim=1).sum().item()
        total += x.size(0)
    return {"top1": correctos / total, "top5": top5_correctos / total}


def main():
    transformacion = transforms.Compose([
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESV),
    ])
    conjunto = datasets.ImageFolder(DIR_DATOS / "prueba", transformacion)
    cargador = DataLoader(conjunto, batch_size=64, shuffle=False, num_workers=2)

    with open(BASE / "entrenamiento" / "clases.json", encoding="utf-8") as fh:
        mapa = json.load(fh)
    n_clases = len(mapa)

    modelo = models.mobilenet_v2(weights=None)
    modelo.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(modelo.classifier[1].in_features, n_clases))
    estado = torch.load(DIR_RESULTADOS / "mejor_modelo.pth", map_location="cpu")

    resultados = {}
    for formato in ("f32", "f16", "i8"):
        modelo.load_state_dict(estado if formato == "f32" else cuantizar(estado, formato))
        resultados[formato] = evaluar(modelo, cargador)
        print(f"[{formato}] {resultados[formato]}")

    with open(DIR_RESULTADOS / "cuantizacion.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2)
    print(f"[OK] Comparación guardada en {DIR_RESULTADOS / 'cuantizacion.json'}")


if __name__ == "__main__":
    main()
