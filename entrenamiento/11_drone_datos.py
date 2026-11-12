"""
11 - Generación del dataset: altitud del dron vs temperatura y presión
======================================================================
Tarea: determinar la ALTURA de un dron a partir de la TEMPERATURA y la
PRESIÓN ATMOSFÉRICA medidas por sus sensores (video de referencia: «Tu
primera red neuronal en Python y TensorFlow», Ringa Tech).

Como no existe una base de datos pública con estas mediciones, se CREAN
datos realistas simulando la atmósfera estándar internacional (ISA):

  - La temperatura desciende con la altura a razón de L = 6,5 °C por km
        T(h) = T_nivel_mar - L·h
  - La presión disminuye con la altura según la fórmula barométrica
        P(h) = P0 · (1 - L·h / T0)^5,2559
    con P0 = 1013,25 hPa y T0 la temperatura (en Kelvin) a nivel del mar.

A las mediciones se les añade el ruido típico de un sensor barométrico
(±0,5 °C y ±0,5 hPa), de modo que la red debe aprender una relación
robusta y no una fórmula exacta.

Salida: entrenamiento/resultados/drone/datos_drone.csv
Uso:    python entrenamiento/11_drone_datos.py
"""

import csv
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
SALIDA = BASE / "entrenamiento" / "resultados" / "drone"
SALIDA.mkdir(parents=True, exist_ok=True)

# ── Constantes de la atmósfera estándar ──────────────────────────────
P0 = 1013.25          # presión a nivel del mar (hPa)
L = 0.0065            # gradiente térmico (°C/m)
EXPONENTE = 5.2559    # g·M / (R·L) de la fórmula barométrica

N_MUESTRAS = 40_000
H_MIN, H_MAX = 0.0, 5000.0        # rango de alturas simuladas (m)
TS_MIN, TS_MAX = -5.0, 38.0       # clima a nivel del mar (°C)
RUIDO_T = 0.5                     # ruido del sensor de temperatura (°C)
RUIDO_P = 0.5                     # ruido del sensor de presión (hPa)
SEMILLA = 2026


def main():
    rng = np.random.default_rng(SEMILLA)

    altura = rng.uniform(H_MIN, H_MAX, N_MUESTRAS)                 # valor real (objetivo)
    ts = rng.uniform(TS_MIN, TS_MAX, N_MUESTRAS)                   # clima a nivel del mar

    temperatura = ts - L * altura + rng.normal(0, RUIDO_T, N_MUESTRAS)      # sensor T (°C)
    t0 = ts + 273.15                                                        # Kelvin a nivel del mar
    presion = P0 * (1 - L * altura / t0) ** EXPONENTE                       # sensor P (hPa)
    presion += rng.normal(0, RUIDO_P, N_MUESTRAS)

    destino = SALIDA / "datos_drone.csv"
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["altura_m", "temperatura_c", "presion_hpa"])
        escritor.writerows(zip(altura.round(3), temperatura.round(3), presion.round(3)))

    print(f"[OK] {N_MUESTRAS} muestras guardadas en {destino}")
    print(f"     Altura:      {altura.min():7.1f} … {altura.max():7.1f} m")
    print(f"     Temperatura: {temperatura.min():7.1f} … {temperatura.max():7.1f} °C")
    print(f"     Presión:     {presion.min():7.1f} … {presion.max():7.1f} hPa")

    # ── Casos de referencia para el simulador (sin ruido) ────────────
    # Se definen por su altitud real y la temperatura que se mediría allí;
    # la presión se calcula con la fórmula barométrica consistente.
    ejemplos = [
        ("Nivel del mar", 10.0, 24.0),
        ("Bogotá (Colombia)", 2640.0, 14.0),
        ("Medellín (Colombia)", 1495.0, 22.0),
        ("Vuelo bajo de dron", 120.0, 23.0),
        ("Zona montañosa", 4000.0, -6.0),
    ]
    casos = []
    for nombre, h, t_med in ejemplos:
        t0 = t_med + L * h                       # temperatura a nivel del mar (K al sumar 273,15)
        p_med = P0 * (1 - L * h / (t0 + 273.15)) ** EXPONENTE
        casos.append({
            "nombre": nombre, "altura": round(h, 1),
            "temperatura": round(t_med, 1), "presion": round(p_med, 1),
        })
        print(f"     Ejemplo {nombre}: h={h:.0f} m -> T={t_med:.1f} °C, P={p_med:.1f} hPa")

    import json
    with open(SALIDA / "ejemplos_drone.json", "w", encoding="utf-8") as fh:
        json.dump(casos, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
