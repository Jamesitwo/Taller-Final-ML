"""
12 - Entrenamiento de la RNA y generación del simulador HTML
============================================================
Tarea: RNA que estima la ALTURA del dron a partir de TEMPERATURA y
PRESIÓN ATMOSFÉRICA (metodología del video «Tu primera red neuronal en
Python y TensorFlow» de Ringa Tech, aplicada a este problema).

La red es un perceptrón multicapa de regresión:
    Entrada (2: temperatura, presión)
      -> Oculta 1 (32 neuronas, tanh)
      -> Oculta 2 (32 neuronas, tanh)
      -> Salida (1: altura en metros)

Después del entrenamiento:
  - se evalúa con datos nunca vistos (MAE, RMSE, R²),
  - los pesos se exportan a JSON,
  - se genera web/tareas/altura_drone.html con la RNA implementada en
    JavaScript puro (sin librerías) para simular la altura en el navegador.

Uso:
    python entrenamiento/12_drone_entrenar.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent.parent
DRONE = BASE / "entrenamiento" / "resultados" / "drone"
WEB_TAREAS = BASE / "web" / "tareas"

SEMILLA = 2026
NEURONAS = 32
EPOCAS = 300
LOTE = 256
LR = 1e-3

torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)
torch.set_num_threads(16)


class RNAAltura(nn.Module):
    """Perceptrón multicapa para regresión (2 -> 32 -> 32 -> 1)."""

    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(2, NEURONAS), nn.Tanh(),
            nn.Linear(NEURONAS, NEURONAS), nn.Tanh(),
            nn.Linear(NEURONAS, 1),
        )

    def forward(self, x):
        return self.red(x)


def main():
    # ── datos ────────────────────────────────────────────────────────
    datos = np.loadtxt(DRONE / "datos_drone.csv", delimiter=",", skiprows=1)
    rng = np.random.default_rng(SEMILLA)
    rng.shuffle(datos)

    n = len(datos)
    n_ent, n_val = int(n * 0.8), int(n * 0.1)
    ent, val, prueba = datos[:n_ent], datos[n_ent:n_ent + n_val], datos[n_ent + n_val:]

    x_ent, y_ent = ent[:, 1:], ent[:, 0:1]
    x_val, y_val = val[:, 1:], val[:, 0:1]
    x_pru, y_pru = prueba[:, 1:], prueba[:, 0:1]

    media_x, desv_x = x_ent.mean(0), x_ent.std(0) + 1e-9
    media_y, desv_y = y_ent.mean(), y_ent.std() + 1e-9

    def normalizar(x):
        return torch.tensor(((x - media_x) / desv_x), dtype=torch.float32)

    def normalizar_y(y):
        return torch.tensor(((y - media_y) / desv_y), dtype=torch.float32)

    # ── entrenamiento ────────────────────────────────────────────────
    modelo = RNAAltura()
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, EPOCAS)

    x_t, y_t = normalizar(x_ent), normalizar_y(y_ent)
    x_v, y_v = normalizar(x_val), normalizar_y(y_val)
    n_lotes = int(np.ceil(len(x_t) / LOTE))

    mejor = float("inf")
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        permutacion = torch.randperm(len(x_t))
        perdida_total = 0.0
        for i in range(n_lotes):
            idx = permutacion[i * LOTE:(i + 1) * LOTE]
            pred = modelo(x_t[idx])
            perdida = criterio(pred, y_t[idx])
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item() * len(idx)
        programador.step()

        modelo.eval()
        with torch.no_grad():
            perdida_val = criterio(modelo(x_v), y_v).item()
        if perdida_val < mejor:
            mejor = perdida_val
            torch.save(modelo.state_dict(), DRONE / "mejor_modelo_drone.pth")
        if epoca % 50 == 0 or epoca == 1:
            print(f"[{epoca:3d}/{EPOCAS}] perdida_ent={perdida_total / len(x_t):.5f}  "
                  f"perdida_val={perdida_val:.5f}", flush=True)

    # ── evaluación sobre el conjunto de prueba ───────────────────────
    modelo.load_state_dict(torch.load(DRONE / "mejor_modelo_drone.pth"))
    modelo.eval()
    with torch.no_grad():
        pred = (modelo(normalizar(x_pru)).numpy() * desv_y + media_y).ravel()
    real = y_pru.ravel()
    error = pred - real
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    r2 = float(1 - np.sum(error ** 2) / np.sum((real - real.mean()) ** 2))
    print(f"\n[PRUEBA] MAE={mae:.2f} m  RMSE={rmse:.2f} m  R²={r2:.5f}")

    # ── exportación de los pesos a JSON (para el navegador) ──────────
    capas = []
    modulos = list(modelo.red)
    for i in (0, 2, 4):
        lineal = modulos[i]
        capas.append({
            "W": lineal.weight.detach().numpy().round(6).tolist(),
            "b": lineal.bias.detach().numpy().round(6).tolist(),
            "activacion": "tanh" if i < 4 else "lineal",
        })
    exportacion = {
        "normalizacion": {
            "entradas": {"media": media_x.round(6).tolist(), "desv": desv_x.round(6).tolist()},
            "salida": {"media": round(float(media_y), 6), "desv": round(float(desv_y), 6)},
        },
        "capas": capas,
    }
    with open(DRONE / "modelo_drone.json", "w", encoding="utf-8") as fh:
        json.dump(exportacion, fh)
    print(f"[OK] Pesos exportados a {DRONE / 'modelo_drone.json'}")

    metricas = {
        "mae_m": round(mae, 2), "rmse_m": round(rmse, 2), "r2": round(r2, 5),
        "muestras_entrenamiento": int(len(x_t)), "muestras_prueba": int(len(x_pru)),
        "neuronas": NEURONAS, "epocas": EPOCAS,
        "ejemplos": json.loads((DRONE / "ejemplos_drone.json").read_text(encoding="utf-8")),
    }

    # ── generación del simulador HTML ────────────────────────────────
    plantilla = (WEB_TAREAS / "plantilla_drone.html").read_text(encoding="utf-8")
    html = (plantilla
            .replace("__MODELO_DRONE_JSON__", json.dumps(exportacion, separators=(",", ":")))
            .replace("__METRICAS_DRONE_JSON__", json.dumps(metricas, ensure_ascii=False, separators=(",", ":"))))
    destino = WEB_TAREAS / "altura_drone.html"
    destino.write_text(html, encoding="utf-8")
    print(f"[OK] Simulador generado en {destino} ({destino.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
