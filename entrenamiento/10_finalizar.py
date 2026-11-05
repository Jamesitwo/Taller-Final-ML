"""
10 - Finalización automática del pipeline
=========================================
Se ejecuta cuando 07_verificar_cuantizacion.py ha terminado:

  1. Elige el formato de cuantización (i8 si no pierde exactitud; si no, f16).
  2. Exporta los pesos para el navegador (03).
  3. Verifica que la implementación JavaScript reproduce a PyTorch (08, Node).
  4. Genera el archivo HTML final autocontenido (06).

Uso:
    python entrenamiento/10_finalizar.py
"""

import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ENTRENA = BASE / "entrenamiento"
RESULTADOS = ENTRENA / "resultados"


def main():
    with open(RESULTADOS / "cuantizacion.json", encoding="utf-8") as fh:
        cuant = json.load(fh)

    base = cuant["f32"]["top1"]
    formato = "i8"
    for candidato in ("i8", "f16"):
        if cuant[candidato]["top1"] >= base - 0.003:   # pierde menos de 0,3 puntos
            formato = candidato
            break
    print(f"[OK] Formato elegido: {formato}  (f32={base:.4f} f16={cuant['f16']['top1']:.4f} i8={cuant['i8']['top1']:.4f})")

    def correr(ejecutable, *args):
        print(f"[...] {ejecutable} {' '.join(args)}", flush=True)
        orden = ([sys.executable, ejecutable, *args] if ejecutable.endswith(".py")
                 else [ejecutable, *args])
        subprocess.run(orden, cwd=BASE, check=True)

    correr(str(ENTRENA / "03_exportar_js.py"), "--c", formato)
    correr("node", str(ENTRENA / "08_verificar_js_node.js"))
    correr(str(ENTRENA / "06_generar_html.py"))
    print("[OK] Pipeline finalizado: web/index.html listo")


if __name__ == "__main__":
    main()
