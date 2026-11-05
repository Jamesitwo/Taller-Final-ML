"""
03 - Exportación del modelo a formato JavaScript (TensorFlow.js sin conversor)
=============================================================================
Como el navegador no puede leer archivos .pth de PyTorch, este script extrae
todos los tensores de la red entrenada y los guarda en un formato sencillo:

  web/modelo/manifiesto.json   ->  arquitectura + ubicación de cada tensor
  web/modelo/pesos.bin         ->  todos los pesos en un solo archivo binario
  web/modelo/verificacion/     ->  caso de prueba para verificar que la
                                   implementación en JavaScript es idéntica

Formatos de cuantización disponibles:
  f16 : pesos de convoluciones/densas en coma flotante de 16 bits
        (la mitad de tamaño, sin pérdida apreciable)
  i8  : pesos de convoluciones/densas en enteros de 8 bits con escala
        por tensor (4 veces más pequeño, pérdida mínima)

Uso:
    python entrenamiento/03_exportar_js.py --c i8
    python entrenamiento/03_exportar_js.py --c f16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import models

BASE = Path(__file__).resolve().parent.parent
DIR_RESULTADOS = BASE / "entrenamiento" / "resultados"
DIR_MODELO = BASE / "web" / "modelo"
DIR_VERIF = DIR_MODELO / "verificacion"

IMG = 224
torch.manual_seed(2026)   # cabeza de clasificación reproducible en modo --preentrenado


def construir_modelo(n_clases: int) -> torch.nn.Module:
    modelo = models.mobilenet_v2(weights=None)
    modelo.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(modelo.classifier[1].in_features, n_clases),
    )
    return modelo


def es_kernel(nombre, forma) -> bool:
    """Son 'kernels' los pesos de convoluciones (4D) y capas densas (2D).
    Los parámetros de normalización por lotes (1D) se conservan en f32."""
    return nombre.endswith(".weight") and len(forma) >= 2


def exportar(modelo, clases, cuantizacion: str):
    DIR_MODELO.mkdir(parents=True, exist_ok=True)
    DIR_VERIF.mkdir(parents=True, exist_ok=True)

    tensores = {}
    datos_bin = bytearray()
    orden = []

    for nombre, tensor in modelo.state_dict().items():
        arr = tensor.detach().cpu().numpy()
        # PyTorch guarda las convoluciones como [salida, entrada, alto, ancho]
        # y TensorFlow.js espera [alto, ancho, entrada, salida].
        # Las convoluciones en profundidad guardan [entrada, 1, alto, ancho].
        if arr.ndim == 4:
            if arr.shape[1] == 1:                       # depthwise
                arr = arr.transpose(2, 3, 0, 1)
            else:                                       # convolución normal
                arr = arr.transpose(2, 3, 1, 0)
            arr = np.ascontiguousarray(arr)
        # alinea cada tensor a 4 bytes: en JavaScript se leen con Float32Array
        while len(datos_bin) % 4 != 0:
            datos_bin += b"\x00"
        entrada = {"shape": list(arr.shape), "offset": len(datos_bin), "dtype": None}

        if cuantizacion == "i8" and es_kernel(nombre, arr.shape):
            escala = float(np.max(np.abs(arr)) / 127.0) or 1.0
            arr_q = np.clip(np.round(arr / escala), -127, 127).astype(np.int8)
            entrada["dtype"] = "i8"
            entrada["escala"] = escala
            datos_bin += arr_q.tobytes()
        elif cuantizacion == "f16" and es_kernel(nombre, arr.shape):
            entrada["dtype"] = "f16"
            datos_bin += arr.astype(np.float16).tobytes()
        else:
            entrada["dtype"] = "f32"
            datos_bin += arr.astype(np.float32).tobytes()

        entrada["bytes"] = len(datos_bin) - entrada["offset"]
        tensores[nombre] = entrada
        orden.append(nombre)

    manifiesto = {
        "formato": cuantizacion,
        "arquitectura": "MobileNetV2",
        "entrada": IMG,
        "normalizacion": {"media": [0.485, 0.456, 0.406], "desv": [0.229, 0.224, 0.225]},
        "clases": clases,
        "orden": orden,
        "tensores": tensores,
    }
    with open(DIR_MODELO / "manifiesto.json", "w", encoding="utf-8") as fh:
        json.dump(manifiesto, fh, ensure_ascii=False, indent=1)
    with open(DIR_MODELO / "pesos.bin", "wb") as fh:
        fh.write(datos_bin)
    print(f"[OK] manifiesto.json + pesos.bin ({len(datos_bin) / 1e6:.2f} MB, formato {cuantizacion})")

    return manifiesto


def generar_caso_verificacion(modelo, clases, manifiesto):
    """Genera una entrada aleatoria fija y la salida esperada de PyTorch."""
    rng = np.random.default_rng(1234)
    entrada = rng.normal(0.0, 1.0, size=(1, IMG, IMG, 3)).astype(np.float32)  # NHWC

    with torch.no_grad():
        x = torch.from_numpy(entrada.transpose(0, 3, 1, 2))  # -> NCHW
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    entrada.tofile(DIR_VERIF / "entrada.bin")
    with open(DIR_VERIF / "salida_esperada.json", "w", encoding="utf-8") as fh:
        json.dump({
            "clases": clases,
            "probabilidades": [float(p) for p in probs],
            "top5": [
                {"clase": clases[i], "probabilidad": float(probs[i])}
                for i in np.argsort(probs)[::-1][:5]
            ],
        }, fh, ensure_ascii=False, indent=1)
    print(f"[OK] Caso de verificación generado en {DIR_VERIF}")
    print("     Top-5 esperado:",
          ", ".join(f"{c['clase']} ({c['probabilidad']:.3f})" for c in
                    json.load(open(DIR_VERIF / 'salida_esperada.json', encoding='utf-8'))['top5']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", choices=["f16", "i8"], default="i8")
    parser.add_argument("--preentrenado", action="store_true",
                        help="usa la MobileNetV2 de ImageNet sin entrenar (para probar la exportación)")
    args = parser.parse_args()

    with open(BASE / "entrenamiento" / "clases.json", encoding="utf-8") as fh:
        mapa = json.load(fh)
    clases = [mapa[str(i)] for i in range(len(mapa))]

    if args.preentrenado:
        modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        n_entradas = modelo.classifier[1].in_features
        modelo.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2), torch.nn.Linear(n_entradas, len(clases)))
        print("[AVISO] Se usa el modelo preentrenado de ImageNet (sin entrenar)")
    else:
        modelo = construir_modelo(len(clases))
        estado = torch.load(DIR_RESULTADOS / "mejor_modelo.pth", map_location="cpu")
        modelo.load_state_dict(estado)
    modelo.eval()

    manifiesto = exportar(modelo, clases, args.c)
    generar_caso_verificacion(modelo, clases, manifiesto)

    # Resumen de la arquitectura para la documentación
    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"[OK] Parámetros totales de la red: {n_params:,}")


if __name__ == "__main__":
    main()
