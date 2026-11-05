"""
05 - Evaluación detallada del modelo entrenado
==============================================
Calcula sobre el conjunto de prueba (nunca visto en el entrenamiento):
  - exactitud top-1 y top-5 (global)
  - exactitud separando imágenes de laboratorio y de campo (fotos con celular)
  - exactitud por especie
  - matriz de confusión (imagen + pares más confundidos)
  - curvas de entrenamiento

Genera:
  entrenamiento/resultados/metricas_detalladas.json
  entrenamiento/resultados/curvas_entrenamiento.png
  entrenamiento/resultados/matriz_confusion.png

Uso:
    python entrenamiento/05_evaluar.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE = Path(__file__).resolve().parent.parent
DIR_DATOS = BASE / "datos" / "procesado"
DIR_RESULTADOS = BASE / "entrenamiento" / "resultados"
IMG = 224
LOTE = 64
MEDIA = [0.485, 0.456, 0.406]
DESV = [0.229, 0.224, 0.225]

torch.set_num_threads(16)


def construir_modelo(n_clases):
    modelo = models.mobilenet_v2(weights=None)
    modelo.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(modelo.classifier[1].in_features, n_clases),
    )
    return modelo


def main():
    with open(BASE / "entrenamiento" / "clases.json", encoding="utf-8") as fh:
        mapa = json.load(fh)
    clases = [mapa[str(i)] for i in range(len(mapa))]

    transformacion = transforms.Compose([
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESV),
    ])
    conjunto = datasets.ImageFolder(DIR_DATOS / "prueba", transformacion)
    cargador = DataLoader(conjunto, batch_size=LOTE, shuffle=False, num_workers=2)

    # fuente (lab/field) de cada archivo del conjunto de prueba
    fuente_de = {}
    with open(BASE / "datos" / "metadatos_imagenes.csv", encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            if fila["parte"] == "prueba":
                fuente_de[Path(fila["archivo"]).stem] = fila["fuente"]

    modelo = construir_modelo(len(clases))
    modelo.load_state_dict(torch.load(DIR_RESULTADOS / "mejor_modelo.pth", map_location="cpu"))
    modelo.eval()

    todas_pred, todas_real, todas_fuente = [], [], []
    with torch.no_grad():
        for x, y in cargador:
            logits = modelo(x)
            pred = logits.topk(5, dim=1).indices.numpy()
            todas_pred.append(pred)
            todas_real.append(y.numpy())

    pred = np.concatenate(todas_pred)
    real = np.concatenate(todas_real)

    # reconstruye la fuente en el orden del DataLoader
    nombres = [Path(ruta).stem for ruta, _ in conjunto.samples]
    fuentes = np.array([fuente_de.get(n, "lab") for n in nombres])

    top1 = float((pred[:, 0] == real).mean())
    top5 = float((pred == real[:, None]).any(axis=1).mean())

    def exactitud(mask):
        if mask.sum() == 0:
            return None
        return {
            "top1": float((pred[mask, 0] == real[mask]).mean()),
            "top5": float((pred[mask] == real[mask][:, None]).any(axis=1).mean()),
            "imagenes": int(mask.sum()),
        }

    por_fuente = {
        "lab": exactitud(fuentes == "lab"),
        "field": exactitud(fuentes == "field"),
    }

    # exactitud por especie
    por_clase = {}
    for i, nombre in enumerate(clases):
        mask = real == i
        if mask.sum():
            por_clase[nombre] = {
                "imagenes": int(mask.sum()),
                "exactitud": float((pred[mask, 0] == real[mask]).mean()),
            }

    # matriz de confusión
    matriz = np.zeros((len(clases), len(clases)), dtype=np.int64)
    for r, p in zip(real, pred[:, 0]):
        matriz[r, p] += 1

    pares = []
    for i in range(len(clases)):
        for j in range(len(clases)):
            if i != j and matriz[i, j] > 0:
                pares.append((clases[i], clases[j], int(matriz[i, j])))
    pares.sort(key=lambda t: -t[2])

    resultados = {
        "prueba": {"top1": top1, "top5": top5, "imagenes": int(len(real))},
        "por_fuente": por_fuente,
        "por_clase": por_clase,
        "pares_confundidos": [
            {"real": a, "predicha": b, "veces": n} for a, b, n in pares[:20]
        ],
        "clases": len(clases),
        "imagen": IMG,
    }
    with open(DIR_RESULTADOS / "metricas_detalladas.json", "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=1)

    # ---------------- curvas de entrenamiento ----------------
    with open(DIR_RESULTADOS / "historial.json", encoding="utf-8") as fh:
        historial = json.load(fh)
    epocas = [h["epoca"] + (0 if h["fase"] == 1 else 0) for h in historial]
    x = range(1, len(historial) + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].plot(x, [h["perdida_ent"] for h in historial], "o-", label="Entrenamiento")
    ejes[0].plot(x, [h["val_perdida"] for h in historial], "s-", label="Validación")
    ejes[0].set_title("Pérdida (entropía cruzada)")
    ejes[1].plot(x, [h["exact_ent"] for h in historial], "o-", label="Entrenamiento")
    ejes[1].plot(x, [h["val_top1"] for h in historial], "s-", label="Validación (top-1)")
    ejes[1].plot(x, [h["val_top5"] for h in historial], "^-", label="Validación (top-5)")
    ejes[1].set_title("Exactitud")
    for e in ejes:
        e.set_xlabel("Época")
        e.grid(alpha=.3)
        e.legend()
        divisor = sum(1 for h in historial if h["fase"] == 1) + 0.5
        e.axvline(divisor, color="gray", ls="--", lw=1)
    ejes[0].text(divisor + 0.1, ejes[0].get_ylim()[0], " ajuste fino", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "curvas_entrenamiento.png", dpi=110)
    plt.close(fig)

    # ---------------- matriz de confusión ----------------
    fig, eje = plt.subplots(figsize=(13, 13))
    im = eje.imshow(matriz, cmap="Blues")
    eje.set_title(f"Matriz de confusión — conjunto de prueba ({len(real)} imágenes)")
    eje.set_xlabel("Especie predicha")
    eje.set_ylabel("Especie real")
    fig.colorbar(im, fraction=.046)
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "matriz_confusion.png", dpi=100)
    plt.close(fig)

    print(f"[PRUEBA] top1={top1:.4f}  top5={top5:.4f}  ({len(real)} imágenes)")
    print(f"[PRUEBA lab]   {por_fuente['lab']}")
    print(f"[PRUEBA field] {por_fuente['field']}")
    print(f"[OK] Resultados en {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
