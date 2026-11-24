"""Construye el dataset: para cada jugada tuya, Stockfish propone K candidatas y se marca cuál jugaste.

Uso:
    python construir_dataset.py irvinlozano rapid [--desde 2024.01.01] [--k 8] [--depth 10] [--max 0]
Salida:
    datos/dataset.csv.gz   (una fila por candidata; 'grupo' = posición, 'elegida' = 1 si fue tu jugada)
"""
import argparse
import io
import multiprocessing as mp
import os
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import pandas as pd

import caracteristicas as C

AQUI = Path(__file__).parent
STOCKFISH = next((AQUI / "motor").rglob("stockfish*.exe"))

_engine = None
_args = None


def _init(args):
    global _engine, _args
    _args = args
    _engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    _engine.configure({"Threads": 1, "Hash": 32})


def procesar_partida(item):
    idx, fecha, pgn = item
    g = chess.pgn.read_game(io.StringIO(pgn))
    usuario = _args.usuario
    yo = chess.WHITE if g.headers.get("White", "").lower() == usuario else chess.BLACK
    limite = chess.engine.Limit(depth=_args.depth)
    filas = []
    board = g.board()
    reloj = {chess.WHITE: None, chess.BLACK: None}
    for nodo in g.mainline():
        move = nodo.move
        if board.turn == yo:
            cands = C.candidatos(_engine, board, _args.k, limite)
            if cands:
                eval_mejor = cands[0][1]
                pos = C.rasgos_posicion(board, board.fullmove_number,
                                        reloj[yo] if reloj[yo] is not None else 600,
                                        reloj[not yo] if reloj[not yo] is not None else 600,
                                        eval_mejor)
                en_top = any(m == move for m, _ in cands)
                grupo = f"{idx}_{board.ply()}"
                for rank, (m, ev) in enumerate(cands):
                    fila = {"grupo": grupo, "partida": idx, "fecha": fecha, "ply": board.ply(),
                            "fen": board.fen(), "uci": m.uci(), "jugada_real": move.uci(),
                            "en_top": int(en_top), "elegida": int(m == move)}
                    fila.update(pos)
                    fila.update(C.rasgos_jugada(board, m, rank, ev, eval_mejor))
                    filas.append(fila)
        clk = nodo.clock()
        if clk is not None:
            reloj[board.turn] = clk
        board.push(move)
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usuario")
    ap.add_argument("ritmo", nargs="?", default="rapid")
    ap.add_argument("--desde", default="2024.01.01")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--max", type=int, default=0, help="limitar # de partidas (0 = todas)")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--salida", default="datos/dataset.csv.gz")
    args = ap.parse_args()
    args.usuario = args.usuario.lower()

    texto = (AQUI / "datos" / f"{args.usuario}_{args.ritmo}.pgn").read_text(encoding="utf-8")
    f = io.StringIO(texto)
    items = []
    idx = 0
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        fecha = g.headers.get("UTCDate") or g.headers.get("Date", "")
        if fecha >= args.desde and g.headers.get("TimeControl") == "600":
            items.append((idx, fecha, str(g)))
        idx += 1
    items.sort(key=lambda x: x[1])
    if args.max:
        items = items[-args.max:]
    print(f"{len(items)} partidas a procesar con {args.procesos} procesos (k={args.k}, depth={args.depth})", flush=True)

    filas = []
    with mp.Pool(args.procesos, initializer=_init, initargs=(args,)) as pool:
        for i, r in enumerate(pool.imap_unordered(procesar_partida, items, chunksize=4), 1):
            filas.extend(r)
            if i % 250 == 0:
                print(f"  {i}/{len(items)} partidas", flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(AQUI / args.salida, index=False, compression="gzip")
    pos = df.drop_duplicates("grupo")
    print(f"Listo: {len(pos)} posiciones, {len(df)} candidatas. Tu jugada estuvo en el top-{args.k} "
          f"de Stockfish en {pos['en_top'].mean():.1%} de las posiciones.")


if __name__ == "__main__":
    main()
