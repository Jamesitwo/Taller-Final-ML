"""Completa manualmente los nombres comunes que Wikidata no pudo resolver
(límites de tasa de la API). Se ejecuta después de 04_metadatos_especies.py."""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RUTA = BASE / "web" / "metadatos_especies.json"

CURADOS = {
    "Acer negundo": ("arce negundo", "box elder"),
    "Acer palmatum": ("arce japonés", "Japanese maple"),
    "Aesculus hippocastamon": ("castaño de Indias", "horse chestnut"),
    "Aesculus pavi": ("falso castaño escarlata", "red buckeye"),
    "Betula jacqemontii": ("abedul del Himalaya", "Himalayan birch"),
    "Broussonettia papyrifera": ("morera del papel", "paper mulberry"),
    "Carpinus caroliniana": ("carpe americano", "American hornbeam"),
    "Cladrastis lutea": ("arce amarillo americano", "American yellowwood"),
    "Diospyros virginiana": ("caqui americano", "American persimmon"),
    "Evodia daniellii": ("evodia coreano", "Korean evodia"),
    "Magnolia soulangiana": ("magnolia lila", "saucer magnolia"),
    "Malus pumila": ("manzano", "apple tree"),
    "Pinus peucea": ("pino de Macedonia", "Macedonian pine"),
    "Platanus acerifolia": ("plátano de sombra", "London plane"),
    "Prunus sargentii": ("cerezo de Sargent", "Sargent's cherry"),
    "Pyrus calleryana": ("peral de Callery", "Callery pear"),
    "Robinia pseudo-acacia": ("acacia blanca", "black locust"),
    "Styrax japonica": ("estoraque japonés", "Japanese snowbell"),
    "Styrax obassia": ("estoraque de hojas grandes", "fragrant snowbell"),
    "Tilia europaea": ("tilo común", "common lime"),
    "Toona sinensis": ("cedro chino", "Chinese mahogany"),
    "Ginkgo biloba": ("ginkgo", "ginkgo"),
    "Prunus yedoensis": ("cerezo de Yedo", "Yoshino cherry"),
    "Sassafras albidum": ("sasafrás", "sassafras"),
    "Quercus montana": ("roble de castaña", "chestnut oak"),
    "Acer ginnala": ("arce de Amur", "Amur maple"),
}

datos = json.loads(RUTA.read_text(encoding="utf-8"))
for especie, (es, en) in CURADOS.items():
    if especie in datos:
        datos[especie]["nombre_es"] = es
        datos[especie]["nombre_en"] = en

RUTA.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
sin = [k for k, v in datos.items() if not v["nombre_es"] and not v["nombre_en"]]
print(f"[OK] {len(CURADOS)} nombres curados aplicados")
print(f"[OK] Especies aún sin nombre: {len(sin)} {sin}")
