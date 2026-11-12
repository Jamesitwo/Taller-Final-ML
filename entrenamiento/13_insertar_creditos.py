"""Inserta el crédito 'Powered by' en las páginas de tareas externas
(boxplot USD/COP y dashboard MiTienda) de forma idempotente."""

from pathlib import Path

CREDITO = (
    '<div style="margin:26px auto 14px;max-width:1100px;text-align:center;'
    'padding:14px 16px;border-top:1px solid rgba(120,140,160,.25);'
    'font-family:system-ui,-apple-system,sans-serif;font-size:.88rem;color:#5a6f86">'
    '&#9889; Powered by <b style="color:#1d6fe0">James Pesca y Erika Julieth Villada</b></div>'
)

ARCHIVOS = [
    Path(__file__).resolve().parent.parent / "web" / "tareas" / "tipo_cambio_usd_cop_boxplot.html",
    Path(__file__).resolve().parent.parent / "web" / "tareas" / "dashboard_mytienda.html",
]

for archivo in ARCHIVOS:
    html = archivo.read_text(encoding="utf-8", errors="ignore")
    if "Powered by" in html:
        print(f"[OK] Ya contiene el crédito: {archivo.name}")
        continue
    cierre = html.lower().rfind("</body>")
    if cierre == -1:
        html = html + "\n" + CREDITO + "\n"
    else:
        html = html[:cierre] + CREDITO + "\n" + html[cierre:]
    archivo.write_text(html, encoding="utf-8")
    print(f"[OK] Crédito insertado en {archivo.name}")
