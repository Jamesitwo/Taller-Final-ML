"""
14 - Algoritmo genético para el Vendedor Viajero (TSP)
=====================================================
Tarea de Algoritmos Genéticos: programar en Python un algoritmo genético
que resuelva el problema del vendedor viajero para no menos de veinte
ciudades, y entregar un aplicativo HTML que permita ingresar el costo de
ir de una ciudad a otra y muestre la ruta óptima.

El algoritmo genético (AG) implementado:
  - Representación: la ruta es una PERMUTACIÓN de las ciudades.
  - Aptitud: menor costo del recorrido cerrado (se vuelve a la ciudad inicial).
  - Selección: por torneo.
  - Cruce: OX1 (order crossover), apto para permutaciones.
  - Mutación: inversión de un tramo de la ruta (estilo 2-opt).
  - Elitismo: los mejores individuos pasan directamente a la siguiente generación.

Validación: se compara con la solución exacta por FUERZA BRUTA en una
instancia pequeña, y luego se resuelve una instancia demo de 22 ciudades
reales de Colombia (distancias por coordenadas geográficas).

Salidas:
  entrenamiento/resultados/tsp/resultado_tsp.json   (resultado + matriz)
  web/tareas/vendedor_viajero.html                  (aplicativo generado)

Uso:
    python entrenamiento/14_tsp_genetico.py
"""

import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # consola Windows

BASE = Path(__file__).resolve().parent.parent
SALIDA = BASE / "entrenamiento" / "resultados" / "tsp"
SALIDA.mkdir(parents=True, exist_ok=True)
WEB_TAREAS = BASE / "web" / "tareas"

SEMILLA = 2026


# =====================================================================
#  ALGORITMO GENÉTICO
# =====================================================================
class AGVendedor:
    """Algoritmo genético para el problema del vendedor viajero."""

    def __init__(self, matriz, tam_poblacion=200, prob_mutacion=0.15,
                 elitismo=4, tam_torneo=5, semilla=None):
        self.matriz = matriz
        self.n = len(matriz)
        self.tam_poblacion = tam_poblacion
        self.prob_mutacion = prob_mutacion
        self.elitismo = elitismo
        self.tam_torneo = tam_torneo
        self.rng = random.Random(semilla)
        self.poblacion = self._poblacion_inicial()
        self.historial = []          # mejor costo por generación
        self.mejor_ruta = None
        self.mejor_costo = float("inf")

    def _poblacion_inicial(self):
        base = list(range(self.n))
        return [self.rng.sample(base, self.n) for _ in range(self.tam_poblacion)]

    def costo(self, ruta):
        """Costo del recorrido cerrado (vuelve a la ciudad de partida)."""
        total = 0.0
        for i in range(self.n):
            total += self.matriz[ruta[i]][ruta[(i + 1) % self.n]]
        return total

    def _seleccion_torneo(self):
        candidatos = self.rng.sample(self.poblacion, self.tam_torneo)
        return min(candidatos, key=self.costo)

    def _cruce_orden(self, padre, madre):
        """OX1: conserva un tramo del padre y completa con el orden de la madre."""
        a, b = sorted(self.rng.sample(range(self.n), 2))
        hijo = [None] * self.n
        hijo[a:b + 1] = padre[a:b + 1]
        relleno = [gen for gen in madre if gen not in hijo]
        idx = 0
        for i in range(self.n):
            if hijo[i] is None:
                hijo[i] = relleno[idx]
                idx += 1
        return hijo

    def _mutar(self, ruta):
        """Invierte un tramo aleatorio de la ruta (mejora local tipo 2-opt)."""
        a, b = sorted(self.rng.sample(range(self.n), 2))
        ruta[a:b + 1] = reversed(ruta[a:b + 1])
        return ruta

    def evolucionar(self, generaciones):
        """Ejecuta N generaciones y devuelve el historial de costos."""
        for _ in range(generaciones):
            self.poblacion.sort(key=self.costo)
            mejor = self.poblacion[0]
            costo_mejor = self.costo(mejor)
            if costo_mejor < self.mejor_costo:
                self.mejor_costo = costo_mejor
                self.mejor_ruta = list(mejor)
            self.historial.append(self.mejor_costo)

            nueva = [list(ind) for ind in self.poblacion[:self.elitismo]]
            while len(nueva) < self.tam_poblacion:
                hijo = self._cruce_orden(self._seleccion_torneo(), self._seleccion_torneo())
                if self.rng.random() < self.prob_mutacion:
                    hijo = self._mutar(hijo)
                nueva.append(hijo)
            self.poblacion = nueva
        return self.historial


# =====================================================================
#  DATOS DE PRUEBA
# =====================================================================
def distancia_haversine(lat1, lon1, lat2, lon2):
    """Distancia en kilómetros entre dos puntos geográficos."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# 22 ciudades de Colombia (latitud, longitud)
CIUDADES_COLOMBIA = {
    "Bogotá": (4.7110, -74.0721),        "Medellín": (6.2442, -75.5812),
    "Cali": (3.4516, -76.5320),          "Barranquilla": (10.9685, -74.7813),
    "Cartagena": (10.3910, -75.4794),    "Bucaramanga": (7.1193, -73.1227),
    "Santa Marta": (11.2408, -74.1990),  "Manizales": (5.0689, -75.5174),
    "Pereira": (4.8133, -75.6961),       "Armenia": (4.5339, -75.6811),
    "Ibagué": (4.4389, -75.2322),        "Cúcuta": (7.8890, -72.4967),
    "Villavicencio": (4.1420, -73.6266), "Neiva": (2.9273, -75.2819),
    "Pasto": (1.2136, -77.2811),         "Popayán": (2.4448, -76.6147),
    "Montería": (8.7500, -75.8833),      "Sincelejo": (9.3047, -75.3978),
    "Valledupar": (10.4631, -73.2532),   "Riohacha": (11.5444, -72.9072),
    "Quibdó": (5.6947, -76.6611),        "Tunja": (5.5353, -73.3678),
}


def matriz_colombia():
    nombres = list(CIUDADES_COLOMBIA)
    n = len(nombres)
    matriz = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = CIUDADES_COLOMBIA[nombres[i]]
                lat2, lon2 = CIUDADES_COLOMBIA[nombres[j]]
                matriz[i][j] = round(distancia_haversine(lat1, lon1, lat2, lon2), 1)
    return nombres, matriz


def validar_con_fuerza_bruta():
    """Comprueba que el AG encuentre la solución exacta en una instancia pequeña."""
    rng = random.Random(7)
    n = 8
    puntos = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    matriz = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matriz[i][j] = math.dist(puntos[i], puntos[j])

    # solución exacta (todas las permutaciones con ciudad inicial fija)
    mejor_exacto = None
    for resto in itertools.permutations(range(1, n)):
        r = (0,) + resto
        c = sum(matriz[r[i]][r[(i + 1) % n]] for i in range(n))
        if mejor_exacto is None or c < mejor_exacto[0]:
            mejor_exacto = (c, r)

    ag = AGVendedor(matriz, tam_poblacion=150, prob_mutacion=0.2,
                    elitismo=4, tam_torneo=5, semilla=SEMILLA)
    ag.evolucionar(600)
    diferencia = abs(ag.mejor_costo - mejor_exacto[0])
    ok = diferencia < 1e-6
    print(f"[VALIDACIÓN] fuerza bruta = {mejor_exacto[0]:.4f}  |  AG = {ag.mejor_costo:.4f}  "
          f"-> {'✅ el AG encontró la solución óptima' if ok else '❌ no coincide'}")
    return {"fuerza_bruta": mejor_exacto[0], "ag": ag.mejor_costo, "coincide": ok}


def main():
    random.seed(SEMILLA)

    # ── 1. validación contra fuerza bruta (8 ciudades) ───────────────
    validacion = validar_con_fuerza_bruta()

    # ── 2. instancia demo: 22 ciudades de Colombia ───────────────────
    nombres, matriz = matriz_colombia()
    print(f"\n[DEMO] {len(nombres)} ciudades de Colombia (distancias por coordenadas)")

    ag = AGVendedor(matriz, tam_poblacion=250, prob_mutacion=0.15,
                    elitismo=5, tam_torneo=5, semilla=SEMILLA)
    t0 = time.time()
    ag.evolucionar(1200)
    duracion = time.time() - t0

    ruta = [nombres[i] for i in ag.mejor_ruta]
    ruta.append(ruta[0])                       # cierra el recorrido
    print(f"[DEMO] Mejor costo: {ag.mejor_costo:,.1f} km en {duracion:.1f} s")
    print(f"[DEMO] Ruta: {' → '.join(ruta)}")

    # referencia: vecino más cercano (heurística simple)
    def vecino_mas_cercano():
        visitados = {0}
        actual, total = 0, 0.0
        for _ in range(len(nombres) - 1):
            sig = min((j for j in range(len(nombres)) if j not in visitados),
                      key=lambda j: matriz[actual][j])
            total += matriz[actual][sig]
            visitados.add(sig)
            actual = sig
        total += matriz[actual][0]
        return total

    vmc = vecino_mas_cercano()
    mejora = (vmc - ag.mejor_costo) / vmc * 100
    print(f"[DEMO] Vecino más cercano: {vmc:,.1f} km → el AG mejoró {mejora:.1f} %")

    # ── 3. exportación de resultados ─────────────────────────────────
    resultado = {
        "ciudades": nombres,
        "matriz": matriz,
        "coordenadas": {nombre: list(CIUDADES_COLOMBIA[nombre]) for nombre in nombres},
        "mejor_ruta": ag.mejor_ruta + [ag.mejor_ruta[0]],
        "mejor_costo": round(ag.mejor_costo, 1),
        "historial": [round(h, 1) for h in ag.historial],
        "vecino_mas_cercano": round(vmc, 1),
        "parametros": {"poblacion": 250, "generaciones": 1200, "mutacion": 0.15,
                       "elitismo": 5, "torneo": 5},
        "validacion": validacion,
        "tiempo_s": round(duracion, 2),
    }
    with open(SALIDA / "resultado_tsp.json", "w", encoding="utf-8") as fh:
        json.dump(resultado, fh, ensure_ascii=False)
    print(f"[OK] Resultado exportado a {SALIDA / 'resultado_tsp.json'}")

    # ── 4. generación del aplicativo HTML ────────────────────────────
    plantilla = (WEB_TAREAS / "plantilla_tsp.html").read_text(encoding="utf-8")
    html = plantilla.replace("__PRESET_TSP_JSON__",
                             json.dumps(resultado, ensure_ascii=False, separators=(",", ":")))
    destino = WEB_TAREAS / "vendedor_viajero.html"
    destino.write_text(html, encoding="utf-8")
    print(f"[OK] Aplicativo generado en {destino} ({destino.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
