"""
01 - Preparación del dataset Leafsnap
====================================
Proyecto: RNA convolucional para reconocimiento de hojas de plantas.

Este script:
  1. Extrae el archivo `leafsnap-dataset.tar` (si aún no se extrajo).
  2. Lee el índice oficial `leafsnap-dataset-images.txt` (30.866 imágenes,
     185 especies de árboles del noreste de EE.UU., fuentes "lab" y "field").
  3. Redimensiona cada imagen conservando la proporción (letterbox) a 256x256
     para que el entrenamiento sea rápido (no decodifica fotos de 10 MP).
  4. Divide en entrenamiento / validación / prueba (70 % / 15 % / 15 %)
     de forma estratificada por especie y fuente.
  5. Genera `clases.json` (índice -> nombre científico) y `metadatos_imagenes.csv`.

Uso:
    python entrenamiento/01_preparar_datos.py
"""

import csv
import json
import random
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DIR_DATOS = BASE / "datos"
TAR = DIR_DATOS / "leafsnap-dataset.tar"
INDICE = DIR_DATOS / "leafsnap-dataset-images.txt"
DIR_EXTRAIDO = DIR_DATOS / "leafsnap"
DIR_SALIDA = DIR_DATOS / "procesado"

TAM = 256          # lado mayor tras el redimensionado (letterbox cuadrado)
COLOR_RELLENO = (128, 128, 128)   # gris neutro para completar el cuadrado
PROPORCIONES = {"entrenamiento": 0.70, "validacion": 0.15, "prueba": 0.15}
SEMILLA = 2026


def extraer_tar():
    """Extrae el tar de Leafsnap solo si hace falta."""
    dir_esperado = DIR_EXTRAIDO / "dataset" / "images"
    if dir_esperado.exists():
        print(f"[OK] Dataset ya extraído en {DIR_EXTRAIDO}")
        return
    if not TAR.exists():
        raise FileNotFoundError(
            f"No se encontró {TAR}. Descárgalo desde el Wayback Machine:\n"
            "  https://web.archive.org/web/20160322210241id_/"
            "http://leafsnap.com/static/dataset/leafsnap-dataset.tar"
        )
    print(f"[...] Extrayendo {TAR.name} ({TAR.stat().st_size / 1e6:.0f} MB)...")
    DIR_EXTRAIDO.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TAR) as tf:
        tf.extractall(DIR_EXTRAIDO, filter="data")
    print("[OK] Extracción terminada")


def leer_indice():
    with open(INDICE, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def procesar_imagen(args):
    """Redimensiona una imagen conservando proporción y la guarda en 256x256."""
    origen, destino = args
    try:
        with Image.open(origen) as im:
            im = im.convert("RGB")
            w, h = im.size
            escala = TAM / max(w, h)
            nw, nh = max(1, round(w * escala)), max(1, round(h * escala))
            im = im.resize((nw, nh), Image.LANCZOS)
            lienzo = Image.new("RGB", (TAM, TAM), COLOR_RELLENO)
            lienzo.paste(im, ((TAM - nw) // 2, (TAM - nh) // 2))
            lienzo.save(destino, "JPEG", quality=92)
        return True
    except Exception as exc:  # imágenes corruptas o ilegibles
        print(f"[AVISO] {origen}: {exc}")
        return False


def dividir(filas):
    """División estratificada por especie + fuente con la semilla fija."""
    rng = random.Random(SEMILLA)
    grupos = {}
    for fila in filas:
        grupos.setdefault((fila["species"], fila["source"]), []).append(fila)

    asignacion = {}
    for clave, items in grupos.items():
        rng.shuffle(items)
        n = len(items)
        n_prueba = max(1, round(n * PROPORCIONES["prueba"]))
        n_valid = max(1, round(n * PROPORCIONES["validacion"]))
        if n_prueba + n_valid >= n:          # grupos muy pequeños
            n_prueba, n_valid = 1, 1
        for i, item in enumerate(items):
            if i < n_prueba:
                parte = "prueba"
            elif i < n_prueba + n_valid:
                parte = "validacion"
            else:
                parte = "entrenamiento"
            asignacion[item["file_id"]] = parte
    return asignacion


def main():
    extraer_tar()
    filas = leer_indice()
    print(f"[OK] Índice leído: {len(filas)} imágenes")

    especies = sorted({f["species"] for f in filas})
    clases = {i: e for i, e in enumerate(especies)}
    with open(BASE / "entrenamiento" / "clases.json", "w", encoding="utf-8") as fh:
        json.dump(clases, fh, ensure_ascii=False, indent=2)
    print(f"[OK] {len(especies)} especies guardadas en entrenamiento/clases.json")

    asignacion = dividir(filas)

    # Prepara rutas de origen/destino
    tareas = []
    metadatos = []
    for fila in filas:
        origen = DIR_EXTRAIDO / fila["image_path"]
        parte = asignacion[fila["file_id"]]
        nombre = f"{fila['file_id']}.jpg"
        destino = DIR_SALIDA / parte / fila["species"].replace(" ", "_") / nombre
        tareas.append((str(origen), str(destino)))
        metadatos.append(
            {
                "file_id": fila["file_id"],
                "especie": fila["species"],
                "fuente": fila["source"],
                "parte": parte,
                "archivo": str(destino.relative_to(DIR_SALIDA)).replace("\\", "/"),
            }
        )

    # Crea directorios
    for _, destino in tareas:
        Path(destino).parent.mkdir(parents=True, exist_ok=True)

    print(f"[...] Procesando {len(tareas)} imágenes a {TAM}x{TAM} (esto tarda unos minutos)")
    with ProcessPoolExecutor(max_workers=8) as pool:
        resultados = list(pool.map(procesar_imagen, tareas, chunksize=64))
    print(f"[OK] Imágenes generadas: {sum(resultados)}/{len(tareas)}")

    with open(DIR_DATOS / "metadatos_imagenes.csv", "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=["file_id", "especie", "fuente", "parte", "archivo"])
        escritor.writeheader()
        escritor.writerows(metadatos)

    conteo = {}
    for m in metadatos:
        conteo[m["parte"]] = conteo.get(m["parte"], 0) + 1
    print("[OK] Metadatos guardados en datos/metadatos_imagenes.csv")
    print(f"[OK] Divisiones: {conteo}")


if __name__ == "__main__":
    main()
