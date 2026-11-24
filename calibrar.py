"""Calibra la temperatura del softmax: ¿con cuánto azar el clon comete errores del mismo tamaño que tú?

Usa las partidas de prueba (las más recientes). Para cada temperatura simula la elección del clon y compara
la pérdida promedio en centipeones contra la tuya real.

Uso:
    python calibrar.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import caracteristicas as C

AQUI = Path(__file__).parent


def main():
    df = pd.read_csv(AQUI / "datos" / "dataset.csv.gz")
    partidas = df.drop_duplicates("partida").sort_values("fecha")["partida"].tolist()
    test = df[df["partida"].isin(set(partidas[int(len(partidas) * 0.85):]))].copy()
    modelo = lgb.Booster(model_file=str(AQUI / "modelo" / "clon.txt"))
    test["score"] = modelo.predict(test[C.COLUMNAS])

    # tu pérdida real, en posiciones donde tu jugada estaba entre las candidatas
    real = test[test["elegida"] == 1]["perdida_cp"].mean()
    rng = np.random.default_rng(0)
    filas = []
    for t in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        perd, acierto = [], []
        for _, g in test[test["en_top"] == 1].groupby("grupo", sort=False):
            s = g["score"].to_numpy()
            if t == 0:
                i = int(np.argmax(s))
            else:
                p = np.exp((s - s.max()) / t)
                i = int(rng.choice(len(s), p=p / p.sum()))
            perd.append(g["perdida_cp"].iat[i])
            acierto.append(g["elegida"].iat[i] == 1)
        filas.append({"temperatura": t, "perdida_cp_clon": round(float(np.mean(perd)), 1),
                      "acierto": round(float(np.mean(acierto)), 3)})
        print(filas[-1], flush=True)
    mejor = min(filas[1:], key=lambda f: abs(f["perdida_cp_clon"] - real))
    print(f"Tu pérdida real: {real:.1f} cp/jugada -> temperatura recomendada: {mejor['temperatura']}")
    (AQUI / "modelo" / "calibracion.json").write_text(
        json.dumps({"perdida_real": round(float(real), 1), "tabla": filas, "recomendada": mejor["temperatura"]},
                   indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
