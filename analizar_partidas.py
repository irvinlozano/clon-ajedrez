"""Resumen rápido de las partidas descargadas: cantidad, Elo, resultados, aperturas.

Uso:
    python analizar_partidas.py irvinlozano rapid
"""
import collections
import datetime as dt
import io
import sys
from pathlib import Path

import chess.pgn


def main(usuario: str, ritmo: str = "rapid") -> None:
    usuario = usuario.lower()
    pgn_path = Path(__file__).parent / "datos" / f"{usuario}_{ritmo}.pgn"
    texto = pgn_path.read_text(encoding="utf-8")
    f = io.StringIO(texto)

    n = 0
    jugadas_mias = 0
    colores = collections.Counter()
    resultados = collections.Counter()
    aperturas = {"white": collections.Counter(), "black": collections.Counter()}
    primeras = {"white": collections.Counter(), "black": collections.Counter()}
    elos = []  # (fecha, elo)
    ritmos = collections.Counter()
    terminaciones = collections.Counter()

    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        h = g.headers
        n += 1
        color = "white" if h.get("White", "").lower() == usuario else "black"
        colores[color] += 1
        res = h.get("Result")
        if res == "1/2-1/2":
            resultados["tablas"] += 1
        elif (res == "1-0") == (color == "white"):
            resultados["gané"] += 1
        else:
            resultados["perdí"] += 1
        elo = h.get("WhiteElo" if color == "white" else "BlackElo")
        fecha = h.get("UTCDate") or h.get("Date")
        if elo and elo.isdigit() and fecha:
            elos.append((fecha, int(elo)))
        ritmos[h.get("TimeControl")] += 1
        term = h.get("Termination", "")
        terminaciones[term.split(" won ")[-1] if " won " in term else term] += 1
        eco_url = h.get("ECOUrl", "")
        nombre = eco_url.rsplit("/", 1)[-1].replace("-", " ") if eco_url else h.get("ECO", "?")
        nombre = " ".join(nombre.split()[:4])
        aperturas[color][nombre] += 1

        movs = list(g.mainline_moves())
        mias = movs[0::2] if color == "white" else movs[1::2]
        jugadas_mias += len(mias)
        b = g.board()
        san = []
        for m in movs[:4]:
            san.append(b.san(m))
            b.push(m)
        primeras[color][" ".join(san[:2] if color == "white" else san[:2])] += 1

    print(f"Partidas: {n}  | jugadas tuyas (posiciones para entrenar): {jugadas_mias}")
    print(f"Color: {dict(colores)}  | Resultados: {dict(resultados)}")
    elos.sort()
    if elos:
        print(f"Primera partida: {elos[0][0]}  Elo {elos[0][1]}  | Última: {elos[-1][0]}  Elo {elos[-1][1]}")
        print(f"Elo máx: {max(e for _, e in elos)}  mín: {min(e for _, e in elos)}")
        por_anio = collections.defaultdict(list)
        for fch, e in elos:
            por_anio[fch[:4]].append(e)
        for a in sorted(por_anio):
            v = por_anio[a]
            print(f"  {a}: {len(v)} partidas, Elo promedio {sum(v) / len(v):.0f}")
    print("Ritmos:", ritmos.most_common(5))
    print("Cómo terminan:", terminaciones.most_common(6))
    for c, nom in (("white", "BLANCAS"), ("black", "NEGRAS")):
        print(f"\nAperturas con {nom}:")
        for k, v in aperturas[c].most_common(8):
            print(f"  {v:5d}  {k}")
        print(f"Primeras 2 jugadas con {nom}:")
        for k, v in primeras[c].most_common(5):
            print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rapid")
