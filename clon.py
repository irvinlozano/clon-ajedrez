"""El bot "clon": elige jugadas como tú.

1. Si la posición está en tu libro de aperturas (la viste >= 3 veces), juega lo que tú sueles jugar,
   con la misma frecuencia.
2. Si no, Stockfish propone K candidatas y el modelo LightGBM les da un puntaje "de qué tanto se parece a ti".
   Se elige con softmax (modo humano: a veces no juega su favorita, como tú) o la de mayor puntaje.
"""
import json
import random
from pathlib import Path

import chess
import chess.engine
import lightgbm as lgb
import numpy as np
import pandas as pd

import caracteristicas as C
from entrenar import clave

AQUI = Path(__file__).parent


def reloj_estimado(jugada_n: int) -> float:
    """En la GUI no hay reloj: se aproxima tu reloj típico en 10 min (~12 s por jugada)."""
    return max(20.0, 600.0 - 12.0 * jugada_n)


class Clon:
    def __init__(self, k: int = 8, depth: int = 10, temperatura: float = 1.0, usar_libro: bool = True):
        self.k, self.temperatura, self.usar_libro = k, temperatura, usar_libro
        self.limite = chess.engine.Limit(depth=depth)
        self.modelo = lgb.Booster(model_file=str(AQUI / "modelo" / "clon.txt"))
        self.libro = json.loads((AQUI / "modelo" / "libro.json").read_text(encoding="utf-8"))
        exe = next((AQUI / "motor").rglob("stockfish*.exe"))
        self.engine = chess.engine.SimpleEngine.popen_uci(str(exe))

    def elegir(self, board: chess.Board) -> dict:
        """Devuelve {'move': Move, 'origen': 'libro'|'modelo', 'opciones': [(uci, prob), ...]}"""
        k = clave(board.fen())
        if self.usar_libro and k in self.libro:
            cuentas = self.libro[k]
            legales = {m.uci() for m in board.legal_moves}
            cuentas = {u: c for u, c in cuentas.items() if u in legales}
            total = sum(cuentas.values())
            # ignora jugadas que casi nunca haces (< 5%): son excepciones, no tu repertorio
            cuentas = {u: c for u, c in cuentas.items() if c / total >= 0.05} if total else {}
            if cuentas:
                total = sum(cuentas.values())
                opciones = sorted(((u, c / total) for u, c in cuentas.items()), key=lambda x: -x[1])
                u = random.choices([o[0] for o in opciones], [o[1] for o in opciones])[0]
                return {"move": chess.Move.from_uci(u), "origen": "libro", "opciones": opciones}

        cands = C.candidatos(self.engine, board, self.k, self.limite)
        eval_mejor = cands[0][1]
        n = board.fullmove_number
        pos = C.rasgos_posicion(board, n, reloj_estimado(n), reloj_estimado(n), eval_mejor)
        filas = []
        for rank, (m, ev) in enumerate(cands):
            f = dict(pos)
            f.update(C.rasgos_jugada(board, m, rank, ev, eval_mejor))
            filas.append(f)
        scores = self.modelo.predict(pd.DataFrame(filas)[C.COLUMNAS])
        if self.temperatura <= 0:
            probs = np.zeros(len(scores))
            probs[int(np.argmax(scores))] = 1.0
        else:
            z = (scores - scores.max()) / self.temperatura
            probs = np.exp(z) / np.exp(z).sum()
        i = int(np.random.choice(len(cands), p=probs))
        opciones = sorted(((m.uci(), float(p)) for (m, _), p in zip(cands, probs)), key=lambda x: -x[1])
        return {"move": cands[i][0], "origen": "modelo", "opciones": opciones}

    def cerrar(self):
        self.engine.quit()
