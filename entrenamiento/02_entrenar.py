"""
02 - Entrenamiento de la RNA convolucional
==========================================
Proyecto: reconocimiento de hojas de plantas (185 especies, dataset Leafsnap).

Arquitectura
------------
MobileNetV2 (preentrenada en ImageNet) como extractor de características
convolucionales + capas de clasificación entrenadas para las 185 especies.

Fases:
  Fase 1 - Se congelan los bloques convolucionales y solo se entrena la
           cabeza de clasificación (rápido, ~12 min/época en CPU).
  Fase 2 - Se descongelan los últimos bloques convolucionales y se hace
           ajuste fino (fine-tuning) con tasa de aprendizaje muy pequeña.

Uso:
    python entrenamiento/02_entrenar.py
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DIR_DATOS = BASE / "datos" / "procesado"
DIR_SALIDA = BASE / "entrenamiento" / "resultados"
DIR_SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 2026
IMG = 224                 # tamaño de entrada de la red
LOTE = 64                 # tamaño de lote
HILOS_DATOS = 2
EPOCAS_CABEZA = 4         # fase 1
EPOCAS_AJUSTE = 3         # fase 2
LR_CABEZA = 1e-3
LR_AJUSTE = 1e-4
PESO_DECAIMIENTO = 1e-4
RELLENO = (128, 128, 128)

MEDIA = [0.485, 0.456, 0.406]     # normalización ImageNet (igual en el navegador)
DESV = [0.229, 0.224, 0.225]

torch.manual_seed(SEMILLA)
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.set_num_threads(16)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def construir_transformaciones():
    train = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.80, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.RandomRotation(180, fill=RELLENO)], p=0.25),
        transforms.RandomRotation(12, fill=RELLENO),
        transforms.ColorJitter(brightness=0.30, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESV),
    ])
    evaluar = transforms.Compose([
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESV),
    ])
    return train, evaluar


def cargar_datos():
    train_tf, eval_tf = construir_transformaciones()
    conjuntos = {
        "entrenamiento": datasets.ImageFolder(DIR_DATOS / "entrenamiento", train_tf),
        "validacion": datasets.ImageFolder(DIR_DATOS / "validacion", eval_tf),
        "prueba": datasets.ImageFolder(DIR_DATOS / "prueba", eval_tf),
    }
    cargadores = {
        "entrenamiento": DataLoader(conjuntos["entrenamiento"], batch_size=LOTE,
                                    shuffle=True, num_workers=HILOS_DATOS,
                                    persistent_workers=HILOS_DATOS > 0),
        "validacion": DataLoader(conjuntos["validacion"], batch_size=LOTE,
                                 shuffle=False, num_workers=HILOS_DATOS,
                                 persistent_workers=HILOS_DATOS > 0),
        "prueba": DataLoader(conjuntos["prueba"], batch_size=LOTE,
                             shuffle=False, num_workers=HILOS_DATOS,
                             persistent_workers=HILOS_DATOS > 0),
    }
    # ImageFolder ordena las carpetas alfabéticamente -> debe coincidir con clases.json
    with open(BASE / "entrenamiento" / "clases.json", encoding="utf-8") as fh:
        mapa_clases = json.load(fh)
    clases_json = [mapa_clases[str(i)] for i in range(len(mapa_clases))]
    clases_folder = [c.replace("_", " ") for c in conjuntos["entrenamiento"].classes]
    assert clases_folder == clases_json, (
        f"Orden de clases distinto:\n folder={clases_folder[:5]}\n json={clases_json[:5]}"
    )
    return conjuntos, cargadores


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def construir_modelo(n_clases: int) -> nn.Module:
    modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    n_entradas = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n_entradas, n_clases),
    )
    return modelo


def congelar_base(modelo):
    for p in modelo.features.parameters():
        p.requires_grad = False


def descongelar_ultimos_bloques(modelo, n_bloques=4):
    """Descongela los últimos n bloques invertidos + convolución final."""
    modulos = list(modelo.features.children())
    for m in modulos[-n_bloques:]:
        for p in m.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def entrenar_epoca(modelo, cargador, criterio, optimizador, dispositivo):
    modelo.classifier.train()
    modelo.features.eval()          # BatchNorm congelado: estadísticas de ImageNet
    perdida_total, correctos, total = 0.0, 0, 0
    inicio = time.time()
    for i, (x, y) in enumerate(cargador):
        x, y = x.to(dispositivo), y.to(dispositivo)
        if all(not p.requires_grad for p in modelo.features.parameters()):
            with torch.no_grad():
                feats = modelo.features(x)
            feats = feats.mean([2, 3])          # promedio global (GAP), igual que en el forward completo
            logits = modelo.classifier(feats)
        else:
            logits = modelo(x)
        perdida = criterio(logits, y)
        optimizador.zero_grad(set_to_none=True)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * x.size(0)
        correctos += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        if (i + 1) % 25 == 0:
            print(f"    paso {i + 1}/{len(cargador)}  perdida={perdida.item():.4f}  "
                  f"exactitud={100 * correctos / total:.1f}%  "
                  f"({(time.time() - inicio) / (i + 1):.1f} s/paso)", flush=True)
    return perdida_total / total, correctos / total


@torch.no_grad()
def evaluar(modelo, cargador, dispositivo):
    modelo.eval()
    criterio = nn.CrossEntropyLoss()
    perdida_total, correctos, total = 0.0, 0, 0
    top5_correctos = 0
    for x, y in cargador:
        x, y = x.to(dispositivo), y.to(dispositivo)
        logits = modelo(x)
        perdida_total += criterio(logits, y).item() * x.size(0)
        pred = logits.topk(5, dim=1).indices
        correctos += (pred[:, 0] == y).sum().item()
        top5_correctos += (pred == y[:, None]).any(dim=1).sum().item()
        total += x.size(0)
    return {
        "perdida": perdida_total / total,
        "top1": correctos / total,
        "top5": top5_correctos / total,
    }


def main():
    dispositivo = "cpu"
    conjuntos, cargadores = cargar_datos()
    n_clases = len(conjuntos["entrenamiento"].classes)
    print(f"[OK] Datos: {len(conjuntos['entrenamiento'])} entrenamiento / "
          f"{len(conjuntos['validacion'])} validación / {len(conjuntos['prueba'])} prueba")
    print(f"[OK] Clases: {n_clases}")

    modelo = construir_modelo(n_clases).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    historial = []

    # ---------------- Fase 1: cabeza de clasificación ----------------
    print("\n=== FASE 1: cabeza de clasificación (base congelada) ===")
    congelar_base(modelo)
    optimizador = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, modelo.parameters()),
        lr=LR_CABEZA, weight_decay=PESO_DECAIMIENTO)
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, EPOCAS_CABEZA)

    mejor_val = 0.0
    for epoca in range(1, EPOCAS_CABEZA + 1):
        t0 = time.time()
        perdida, exact = entrenar_epoca(modelo, cargadores["entrenamiento"], criterio, optimizador, dispositivo)
        val = evaluar(modelo, cargadores["validacion"], dispositivo)
        programador.step()
        historial.append({"fase": 1, "epoca": epoca, "perdida_ent": perdida,
                          "exact_ent": exact, **{f"val_{k}": v for k, v in val.items()}})
        print(f"[F1 {epoca}/{EPOCAS_CABEZA}] ent={exact:.3f} val_top1={val['top1']:.3f} "
              f"val_top5={val['top5']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if val["top1"] > mejor_val:
            mejor_val = val["top1"]
            torch.save(modelo.state_dict(), DIR_SALIDA / "mejor_modelo.pth")

    # ---------------- Fase 2: ajuste fino ----------------
    print("\n=== FASE 2: ajuste fino de los últimos bloques ===")
    modelo.load_state_dict(torch.load(DIR_SALIDA / "mejor_modelo.pth"))
    descongelar_ultimos_bloques(modelo, n_bloques=4)
    optimizador = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, modelo.parameters()),
        lr=LR_AJUSTE, weight_decay=PESO_DECAIMIENTO)
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, EPOCAS_AJUSTE)

    for epoca in range(1, EPOCAS_AJUSTE + 1):
        t0 = time.time()
        perdida, exact = entrenar_epoca(modelo, cargadores["entrenamiento"], criterio, optimizador, dispositivo)
        val = evaluar(modelo, cargadores["validacion"], dispositivo)
        programador.step()
        historial.append({"fase": 2, "epoca": epoca, "perdida_ent": perdida,
                          "exact_ent": exact, **{f"val_{k}": v for k, v in val.items()}})
        print(f"[F2 {epoca}/{EPOCAS_AJUSTE}] ent={exact:.3f} val_top1={val['top1']:.3f} "
              f"val_top5={val['top5']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if val["top1"] > mejor_val:
            mejor_val = val["top1"]
            torch.save(modelo.state_dict(), DIR_SALIDA / "mejor_modelo.pth")

    # ---------------- Evaluación final sobre prueba ----------------
    modelo.load_state_dict(torch.load(DIR_SALIDA / "mejor_modelo.pth"))
    resultado_prueba = evaluar(modelo, cargadores["prueba"], dispositivo)
    print(f"\n[PRUEBA] top1={resultado_prueba['top1']:.4f}  top5={resultado_prueba['top5']:.4f}")

    with open(DIR_SALIDA / "historial.json", "w", encoding="utf-8") as fh:
        json.dump(historial, fh, indent=2)
    with open(DIR_SALIDA / "metricas.json", "w", encoding="utf-8") as fh:
        json.dump({"prueba": resultado_prueba, "mejor_val_top1": mejor_val,
                   "clases": n_clases, "imagen": IMG}, fh, indent=2)
    print(f"[OK] Resultados guardados en {DIR_SALIDA}")


if __name__ == "__main__":
    main()
