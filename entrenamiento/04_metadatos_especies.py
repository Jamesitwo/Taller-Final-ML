"""
04 - Metadatos de las especies (Wikidata / Wikipedia)
=====================================================
Descarga para cada una de las 185 especies del dataset Leafsnap:
  - nombre científico
  - nombre común en español e inglés
  - familia botánica
  - género

Los datos se guardan en web/metadatos_especies.json para mostrar una ficha
completa cuando la red neuronal identifica una hoja.

Uso:
    python entrenamiento/04_metadatos_especies.py
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
AGENTE = "ProyectoRNAHojas/1.0 (proyecto educativo de machine learning)"

FAMILIAS = {
    "Abies": "Pinaceae", "Acer": "Sapindaceae", "Aesculus": "Sapindaceae",
    "Ailanthus": "Simaroubaceae", "Albizia": "Fabaceae", "Amelanchier": "Rosaceae",
    "Asimina": "Annonaceae", "Betula": "Betulaceae", "Broussonettia": "Moraceae",
    "Carpinus": "Betulaceae", "Carya": "Juglandaceae", "Castanea": "Fagaceae",
    "Catalpa": "Bignoniaceae", "Cedrus": "Pinaceae", "Celtis": "Cannabaceae",
    "Cercidiphyllum": "Cercidiphyllaceae", "Cercis": "Fabaceae",
    "Chamaecyparis": "Cupressaceae", "Chionanthus": "Oleaceae",
    "Cladrastis": "Fabaceae", "Cornus": "Cornaceae", "Corylus": "Betulaceae",
    "Crataegus": "Rosaceae", "Cryptomeria": "Cupressaceae", "Diospyros": "Ebenaceae",
    "Eucommia": "Eucommiaceae", "Evodia": "Rutaceae", "Fagus": "Fagaceae",
    "Ficus": "Moraceae", "Fraxinus": "Oleaceae", "Ginkgo": "Ginkgoaceae",
    "Gleditsia": "Fabaceae", "Gymnocladus": "Fabaceae", "Halesia": "Styracaceae",
    "Ilex": "Aquifoliaceae", "Juglans": "Juglandaceae", "Juniperus": "Cupressaceae",
    "Koelreuteria": "Sapindaceae", "Larix": "Pinaceae", "Liquidambar": "Altingiaceae",
    "Liriodendron": "Magnoliaceae", "Maclura": "Moraceae", "Magnolia": "Magnoliaceae",
    "Malus": "Rosaceae", "Metasequoia": "Cupressaceae", "Morus": "Moraceae",
    "Nyssa": "Nyssaceae", "Ostrya": "Betulaceae", "Oxydendrum": "Ericaceae",
    "Paulownia": "Paulowniaceae", "Phellodendron": "Rutaceae", "Picea": "Pinaceae",
    "Pinus": "Pinaceae", "Platanus": "Platanaceae", "Populus": "Salicaceae",
    "Prunus": "Rosaceae", "Ptelea": "Rutaceae", "Pyrus": "Rosaceae",
    "Quercus": "Fagaceae", "Robinia": "Fabaceae", "Salix": "Salicaceae",
    "Sassafras": "Lauraceae", "Staphylea": "Staphyleaceae", "Stewartia": "Theaceae",
    "Styrax": "Styracaceae", "Syringa": "Oleaceae", "Taxodium": "Cupressaceae",
    "Tilia": "Malvaceae", "Toona": "Meliaceae", "Tsuga": "Pinaceae",
    "Ulmus": "Ulmaceae", "Zelkova": "Ulmaceae",
}

TRADUCCION_FAMILIA = {
    "Pinaceae": "Familia de los pinos", "Sapindaceae": "Familia del arce",
    "Fabaceae": "Familia de las leguminosas", "Rosaceae": "Familia de las rosas",
    "Betulaceae": "Familia del abedul", "Fagaceae": "Familia del roble",
    "Moraceae": "Familia de la mora", "Juglandaceae": "Familia del nogal",
    "Cupressaceae": "Familia de los cipreses", "Oleaceae": "Familia del olivo",
    "Magnoliaceae": "Familia de la magnolia", "Salicaceae": "Familia del sauce",
    "Ulmaceae": "Familia del olmo", "Rutaceae": "Familia de los cítricos",
    "Annonaceae": "Familia de la chirimoya", "Aquifoliaceae": "Familia del acebo",
    "Bignoniaceae": "Familia de la catalpa", "Cannabaceae": "Familia del cáñamo",
    "Cornaceae": "Familia del cornejo", "Ebenaceae": "Familia del ébano",
    "Ericaceae": "Familia del brezo", "Lauraceae": "Familia del laurel",
    "Malvaceae": "Familia del tilo", "Meliaceae": "Familia de la caoba",
    "Nyssaceae": "Familia del tupelo", "Platanaceae": "Familia del plátano",
    "Simaroubaceae": "Familia del ailanto", "Styracaceae": "Familia del estorace",
    "Theaceae": "Familia del té", "Altingiaceae": "Familia del liquidámbar",
    "Cercidiphyllaceae": "Familia del katsura", "Eucommiaceae": "Familia del eucommia",
    "Ginkgoaceae": "Familia del ginkgo", "Paulowniaceae": "Familia de la paulonia",
    "Staphyleaceae": "Familia del falso pistacho", "Cupressaceae ": "Familia de los cipreses",
}


def api(url, params):
    """Consulta a la API de Wikimedia con manejo simple de errores."""
    params = dict(params, format="json")
    destino = url + "?" + urllib.parse.urlencode(params)
    peticion = urllib.request.Request(destino, headers={"User-Agent": AGENTE})
    for intento in range(6):
        try:
            with urllib.request.urlopen(peticion, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            if intento == 5:
                print(f"[AVISO] {destino}: {exc}")
                return {}
            time.sleep(2.0 * (intento + 1))   # retroceso exponencial (429)
    return {}


def buscar_en_wikidata(nombre):
    datos = api("https://www.wikidata.org/w/api.php", {
        "action": "wbsearchentities", "search": nombre,
        "language": "es", "uselang": "es", "type": "item", "limit": "5",
    })
    candidatos = datos.get("search", [])
    # Elige el candidato cuya etiqueta coincide con el nombre científico
    for cand in candidatos:
        if cand.get("label", "").strip().lower() == nombre.lower():
            return cand["id"]
    for cand in candidatos:
        desc = (cand.get("description") or "").lower()
        if "especie" in desc or "taxón" in desc or "planta" in desc or "árbol" in desc:
            return cand["id"]
    return candidatos[0]["id"] if candidatos else None


def detalles_wikidata(qid):
    datos = api("https://www.wikidata.org/w/api.php", {
        "action": "wbgetentities", "ids": qid,
        "props": "labels|descriptions|sitelinks|claims",
        "languages": "es|en", "sitefilter": "eswiki|enwiki",
    })
    return datos.get("entities", {}).get(qid, {})


def nombre_comun(entidad, cientifico=""):
    """Nombre común: preferencia a la declaración P1843 (nombre común)."""
    nombres = {"es": "", "en": ""}
    for reclamo in entidad.get("claims", {}).get("P1843", []):
        try:
            valor = reclamo["mainsnak"]["datavalue"]["value"]["text"]
            idioma = reclamo["mainsnak"]["datavalue"]["value"]["language"]
            if idioma in nombres and not nombres[idioma]:
                nombres[idioma] = valor
        except (KeyError, TypeError):
            continue

    def util(titulo):
        titulo = titulo.strip()
        return titulo and titulo.lower() != cientifico.lower()

    enlaces = entidad.get("sitelinks", {})
    if not nombres["es"] and "eswiki" in enlaces and util(enlaces["eswiki"]["title"]):
        nombres["es"] = enlaces["eswiki"]["title"]
    if not nombres["en"] and "enwiki" in enlaces and util(enlaces["enwiki"]["title"]):
        nombres["en"] = enlaces["enwiki"]["title"]
    return nombres


def main():
    with open(BASE / "entrenamiento" / "clases.json", encoding="utf-8") as fh:
        mapa = json.load(fh)
    especies = [mapa[str(i)] for i in range(len(mapa))]

    salida = {}
    for i, nombre in enumerate(especies, 1):
        genero = nombre.split()[0]
        info = {
            "nombre_cientifico": nombre,
            "genero": genero,
            "familia": FAMILIAS.get(genero, ""),
            "familia_es": TRADUCCION_FAMILIA.get(FAMILIAS.get(genero, ""), ""),
            "nombre_es": "",
            "nombre_en": "",
        }
        qid = buscar_en_wikidata(nombre)
        if qid:
            entidad = detalles_wikidata(qid)
            nombres = nombre_comun(entidad, nombre)
            info["nombre_es"] = nombres["es"]
            info["nombre_en"] = nombres["en"]
            info["wikidata"] = qid
        salida[nombre] = info
        print(f"[{i}/{len(especies)}] {nombre} -> {info['nombre_es'] or info['nombre_en'] or '(sin nombre común)'}", flush=True)
        time.sleep(0.35)

    # Pasada de reparación: reintenta las especies que quedaron sin nombre común
    pendientes = [n for n, d in salida.items() if not d["nombre_es"] and not d["nombre_en"]]
    if pendientes:
        print(f"[...] Reparando {len(pendientes)} especies sin nombre común")
        for nombre in pendientes:
            time.sleep(1.0)
            qid = buscar_en_wikidata(nombre)
            if not qid:
                continue
            entidad = detalles_wikidata(qid)
            nombres = nombre_comun(entidad, nombre)
            salida[nombre]["nombre_es"] = nombres["es"]
            salida[nombre]["nombre_en"] = nombres["en"]
            print(f"    {nombre} -> {nombres['es'] or nombres['en'] or '(sin nombre común)'}", flush=True)

    destino = BASE / "web" / "metadatos_especies.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False, indent=1)
    print(f"[OK] Metadatos guardados en {destino}")


if __name__ == "__main__":
    main()
