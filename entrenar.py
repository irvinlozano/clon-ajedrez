"""Entrena el modelo "clon": entre las candidatas de Stockfish, ¿cuál jugarías tú?

Usa LightGBM en modo ranking (LambdaRank): cada posición es un grupo y la jugada que hiciste es la relevante.
Separa por fecha: entrena con tus partidas viejas y evalúa con las más recientes (que el modelo nunca vio).

Uso:
    python entrenar.py [--datos datos/dataset.csv.gz] [--test 0.15]
Salida:
    modelo/clon.txt, modelo/libro.json, modelo/metricas.json, modelo/importancia.png
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import caracteristicas as C

AQUI = Path(__file__).parent
MIN_LIBRO = 3  # veces que debiste ver la posición para usar el libro


def clave(fen: str) -> str:
    return " ".join(fen.split()[:4])  # sin contadores de jugadas


def construir_libro(df: pd.DataFrame) -> dict:
    pos = df.drop_duplicates("grupo")
    libro = defaultdict(Counter)
    for fen, mv in zip(pos["fen"], pos["jugada_real"]):
        libro[clave(fen)][mv] += 1
    return {k: dict(v) for k, v in libro.items() if sum(v.values()) >= MIN_LIBRO}


def evaluar(test: pd.DataFrame, scores: np.ndarray, libro: dict) -> dict:
    t = test.assign(score=scores)
    res = defaultdict(list)
    for _, g in t.groupby("grupo", sort=False):
        real = g["jugada_real"].iat[0]
        ply = g["ply"].iat[0]
        fase = "apertura (jugadas 1-10)" if ply < 20 else ("medio juego (11-30)" if ply < 60 else "final (31+)")
        orden = g.sort_values("score", ascending=False)
        pred = orden["uci"].iat[0]
        top3 = set(orden["uci"].iloc[:3])
        sf = g.loc[g["rank_sf"] == 0, "uci"].iat[0]
        k = clave(g["fen"].iat[0])
        en_libro = k in libro
        pred_clon = max(libro[k], key=libro[k].get) if en_libro else pred
        res["modelo"].append(pred == real)
        res["modelo_top3"].append(real in top3)
        res["clon (libro+modelo)"].append(pred_clon == real)
        res["stockfish"].append(sf == real)
        res["azar entre candidatas"].append(g["elegida"].sum() / len(g))
        res["techo (tu jugada estaba en el top)"].append(bool(g["en_top"].iat[0]))
        res["en_libro"].append(en_libro)
        res["fase"].append(fase)
        res["perdida_clon"].append(orden["perdida_cp"].iat[0])
        if g["en_top"].iat[0]:
            res["perdida_real"].append(g.loc[g["elegida"] == 1, "perdida_cp"].iat[0])
            res["perdida_clon_entop"].append(orden["perdida_cp"].iat[0])

    m = {k: float(np.mean(res[k])) for k in
         ["modelo", "modelo_top3", "clon (libro+modelo)", "stockfish", "azar entre candidatas",
          "techo (tu jugada estaba en el top)", "en_libro"]}
    fases = pd.DataFrame({"fase": res["fase"], "modelo": res["modelo"], "clon": res["clon (libro+modelo)"],
                          "stockfish": res["stockfish"]})
    m["por_fase"] = fases.groupby("fase").mean().round(3).to_dict(orient="index")
    m["perdida_cp_promedio"] = {
        "tu (real)": float(np.mean(res["perdida_real"])),
        "clon (mismas posiciones)": float(np.mean(res["perdida_clon_entop"])),
        "stockfish": 0.0,
    }
    m["posiciones_test"] = len(res["modelo"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="datos/dataset.csv.gz")
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--hilos", type=int, default=-1, help="hilos de LightGBM (-1 = todos)")
    args = ap.parse_args()

    df =pd.read_csv(AQUI / args.datos)
    partidas = df.drop_duplicates("partida").sort_values("fecha")["partida"].tolist()
    corte = int(len(partidas) * (1 - args.test))
    train_ids = set(partidas[:corte])
    train = df[df["partida"].isin(train_ids)]
    test = df[~df["partida"].isin(train_ids)]
    print(f"Entrenamiento: {train['partida'].nunique()} partidas | prueba: {test['partida'].nunique()} partidas "
          f"(desde {test['fecha'].min()})")

    # Para entrenar el ranking solo sirven posiciones donde tu jugada está entre las candidatas.
    tr = train[train["en_top"] == 1]
    # validación interna: último 10% del entrenamiento
    ids_tr = tr.drop_duplicates("partida").sort_values("fecha")["partida"].tolist()
    c2 = int(len(ids_tr) * 0.9)
    fit, val = tr[tr["partida"].isin(ids_tr[:c2])], tr[tr["partida"].isin(ids_tr[c2:])]

    def grupos(d):
        return d.groupby("grupo", sort=False).size().to_numpy()

    modelo = lgb.LGBMRanker(objective="lambdarank", n_estimators=2000, learning_rate=0.05, num_leaves=63,
                            min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            n_jobs=args.hilos, verbose=-1)
    modelo.fit(fit[C.COLUMNAS], fit["elegida"], group=grupos(fit),
               eval_X=(val[C.COLUMNAS],), eval_y=(val["elegida"],), eval_group=[grupos(val)], eval_at=[1],
               callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"Árboles usados: {modelo.best_iteration_}")

    libro = construir_libro(train)
    metricas = evaluar(test, modelo.predict(test[C.COLUMNAS], num_iteration=modelo.best_iteration_), libro)

    out = AQUI / "modelo"
    out.mkdir(exist_ok=True)
    modelo.booster_.save_model(str(out / "clon.txt"), num_iteration=modelo.best_iteration_)
    (out / "libro.json").write_text(json.dumps(libro), encoding="utf-8")
    (out / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    imp = pd.Series(modelo.booster_.feature_importance("gain"), index=C.COLUMNAS).sort_values()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = (imp / imp.sum()).plot.barh(figsize=(7, 7), color="#4a7bd0")
        ax.set_title("¿Qué mira el modelo para jugar como tú?")
        ax.set_xlabel("importancia (ganancia relativa)")
        plt.tight_layout()
        plt.savefig(out / "importancia.png", dpi=130)
    except ImportError:
        pass

    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    print("Top características:", list(imp.index[::-1][:8]))


if __name__ == "__main__":
    main()
