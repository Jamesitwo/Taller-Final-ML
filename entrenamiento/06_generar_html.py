"""
06 - Generación del archivo HTML final (autocontenido)
======================================================
Construye el entregable: un ÚNICO archivo HTML con todo embebido:
  - la implementación de la RNA convolucional (MobileNetV2 en TensorFlow.js)
  - los pesos del modelo entrenado (en base64)
  - los metadatos de las 185 especies
  - las métricas y la gráfica de entrenamiento
  - imágenes de ejemplo para probar la aplicación

También genera una versión "ligera" (web/index-ligero.html + web/modelo/*)
útil para publicar en GitHub Pages con carga más rápida.

Uso:
    python entrenamiento/06_generar_html.py
"""

import base64
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
WEB = BASE / "web"
RESULTADOS = BASE / "entrenamiento" / "resultados"

ESPECIES_EJEMPLO = [
    "Acer rubrum", "Quercus rubra", "Ginkgo biloba",
    "Betula alleghaniensis", "Pinus strobus", "Liriodendron tulipifera",
]


def cargar_json(ruta, defecto=None):
    if not ruta.exists():
        return defecto
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


def a_js(obj):
    """JSON seguro para incrustar dentro de una etiqueta <script>."""
    texto = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return texto.replace("</", "<\\/")


def construir_ejemplos():
    """Miniaturas de ejemplo del conjunto de prueba (imágenes nunca vistas)."""
    ejemplos = []
    metadatos = cargar_json(WEB / "metadatos_especies.json", {})
    procesado = BASE / "datos" / "procesado"
    for especie in ESPECIES_EJEMPLO:
        carpeta = especie.replace(" ", "_")
        archivo = None
        for parte in ("prueba", "validacion", "entrenamiento"):
            ruta_dir = procesado / parte / carpeta
            if ruta_dir.exists():
                candidatos = sorted(ruta_dir.glob("*.jpg"))
                if candidatos:
                    archivo = candidatos[0]
                    break
        if archivo is None:
            continue
        datos = base64.b64encode(archivo.read_bytes()).decode("ascii")
        info = metadatos.get(especie, {})
        ejemplos.append({
            "clase": especie,
            "nombre": info.get("nombre_es") or info.get("nombre_en") or especie,
            "imagen": f"data:image/jpeg;base64,{datos}",
        })
    return ejemplos


def main():
    plantilla = (WEB / "plantilla.html").read_text(encoding="utf-8")
    codigo_red = (WEB / "red-neuronal.js").read_text(encoding="utf-8")

    manifiesto = cargar_json(WEB / "modelo" / "manifiesto.json")
    pesos = (WEB / "modelo" / "pesos.bin").read_bytes()
    metadatos = cargar_json(WEB / "metadatos_especies.json", {})
    metricas_det = cargar_json(RESULTADOS / "metricas_detalladas.json", {})
    metricas_ent = cargar_json(RESULTADOS / "metricas.json", {})
    curvas = (RESULTADOS / "curvas_entrenamiento.png").read_bytes() \
        if (RESULTADOS / "curvas_entrenamiento.png").exists() else None

    # parámetros de la red (solo tensores entrenables: pesos y sesgos)
    parametros = 0
    for nombre in manifiesto["orden"]:
        if nombre.endswith(".weight") or nombre.endswith(".bias"):
            dims = manifiesto["tensores"][nombre]["shape"]
            n = 1
            for d in dims:
                n *= d
            parametros += n
    manifiesto["parametros"] = f"{parametros:,}".replace(",", " ")

    metricas = {
        "top1": (metricas_det.get("prueba") or metricas_ent.get("prueba", {})).get("top1"),
        "top5": (metricas_det.get("prueba") or metricas_ent.get("prueba", {})).get("top5"),
        "top1_campo": (metricas_det.get("por_fuente", {}).get("field") or {}).get("top1"),
        "top1_lab": (metricas_det.get("por_fuente", {}).get("lab") or {}).get("top1"),
        "pares_confundidos": (metricas_det.get("pares_confundidos") or [])[:8],
        "curvas_b64": f"data:image/png;base64,{base64.b64encode(curvas).decode('ascii')}" if curvas else "",
    }

    ejemplos = construir_ejemplos()
    print(f"[OK] {len(ejemplos)} imágenes de ejemplo embebidas")

    reemplazos = {
        "__MANIFIESTO_JSON__": a_js(manifiesto),
        "__PESOS_BASE64__": base64.b64encode(pesos).decode("ascii"),
        "__METADATOS_ESPECIES_JSON__": a_js(metadatos),
        "__METRICAS_JSON__": a_js(metricas),
        "__EJEMPLOS_JSON__": a_js(ejemplos),
        "__RED_JS__": codigo_red,
    }

    # ---------- versión autocontenida (entregable) ----------
    html = plantilla
    for token, valor in reemplazos.items():
        html = html.replace(token, valor)
    destino = WEB / "index.html"
    destino.write_text(html, encoding="utf-8")
    print(f"[OK] {destino}  ({destino.stat().st_size / 1e6:.2f} MB) — archivo único autocontenido")

    # ---------- versión ligera (modelo en archivos aparte) ----------
    html_ligero = plantilla
    for token, valor in reemplazos.items():
        html_ligero = html_ligero.replace(token, "" if token == "__PESOS_BASE64__" else valor)
    (WEB / "index-ligero.html").write_text(html_ligero, encoding="utf-8")
    print(f"[OK] {WEB / 'index-ligero.html'} + {WEB / 'modelo'} — versión ligera")

    # ---------- arnés de verificación numérica ----------
    if (WEB / "plantilla_verificacion.html").exists():
        verif = (WEB / "plantilla_verificacion.html").read_text(encoding="utf-8")
        verif = verif.replace("__RED_JS__", codigo_red)
        (WEB / "verificacion.html").write_text(verif, encoding="utf-8")
        print(f"[OK] {WEB / 'verificacion.html'} — comparación JavaScript vs PyTorch")


if __name__ == "__main__":
    main()
