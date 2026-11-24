"""Características (features) de una jugada candidata en una posición.

Se usan igual al construir el dataset y al jugar, para que el modelo vea lo mismo en los dos casos.
"""
import chess
import chess.engine

VALOR = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
CLIP = 1000  # centipeones

COLUMNAS = [
    # de la posición
    "jugada_n", "mi_reloj", "reloj_rival", "material_mio", "material_rival", "dif_material",
    "n_legales", "en_jaque", "eval_mejor",
    # de la jugada candidata
    "rank_sf", "perdida_cp", "eval_tras", "es_captura", "valor_capturado", "pieza",
    "es_jaque", "es_enroque", "es_promocion", "avance", "dist_centro", "dist_rey_rival",
    "queda_colgada", "es_recaptura", "mueve_dama", "cambia_damas",
]


def material(board: chess.Board, color: bool) -> int:
    return sum(VALOR[p.piece_type] for p in board.piece_map().values() if p.color == color)


def score_cp(score: chess.engine.PovScore, color: bool) -> int:
    s = score.pov(color).score(mate_score=10000)
    return max(-CLIP, min(CLIP, s))


def rasgos_posicion(board: chess.Board, jugada_n: int, mi_reloj: float, reloj_rival: float, eval_mejor: int) -> dict:
    yo = board.turn
    mm, mr = material(board, yo), material(board, not yo)
    return {
        "jugada_n": jugada_n,
        "mi_reloj": mi_reloj,
        "reloj_rival": reloj_rival,
        "material_mio": mm,
        "material_rival": mr,
        "dif_material": mm - mr,
        "n_legales": board.legal_moves.count(),
        "en_jaque": int(board.is_check()),
        "eval_mejor": eval_mejor,
    }


def rasgos_jugada(board: chess.Board, move: chess.Move, rank_sf: int, eval_tras: int, eval_mejor: int) -> dict:
    yo = board.turn
    pieza = board.piece_type_at(move.from_square)
    capturada = board.piece_type_at(move.to_square)
    if board.is_en_passant(move):
        capturada = chess.PAWN
    ultima = board.move_stack[-1] if board.move_stack else None
    rey_rival = board.king(not yo)
    dama_rival_antes = len(board.pieces(chess.QUEEN, not yo))

    fila_desde = chess.square_rank(move.from_square)
    fila_hasta = chess.square_rank(move.to_square)
    avance = (fila_hasta - fila_desde) if yo == chess.WHITE else (fila_desde - fila_hasta)
    f, r = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    dist_centro = max(abs(f - 3.5), abs(r - 3.5))

    b2 = board.copy(stack=False)
    es_jaque = int(board.gives_check(move))
    b2.push(move)
    # ¿la pieza movida queda atacada y sin defensa? (aproxima un "regalo" de pieza)
    atacada = b2.is_attacked_by(not yo, move.to_square)
    defendida = b2.is_attacked_by(yo, move.to_square)
    queda_colgada = int(atacada and not defendida and pieza != chess.KING)
    cambia_damas = int(capturada == chess.QUEEN and pieza == chess.QUEEN and dama_rival_antes == 1)

    return {
        "rank_sf": rank_sf,
        "perdida_cp": max(0, eval_mejor - eval_tras),
        "eval_tras": eval_tras,
        "es_captura": int(capturada is not None),
        "valor_capturado": VALOR.get(capturada, 0) if capturada else 0,
        "pieza": pieza,
        "es_jaque": es_jaque,
        "es_enroque": int(board.is_castling(move)),
        "es_promocion": int(move.promotion is not None),
        "avance": avance,
        "dist_centro": dist_centro,
        "dist_rey_rival": chess.square_distance(move.to_square, rey_rival) if rey_rival is not None else 0,
        "queda_colgada": queda_colgada,
        "es_recaptura": int(ultima is not None and ultima.to_square == move.to_square and capturada is not None),
        "mueve_dama": int(pieza == chess.QUEEN),
        "cambia_damas": cambia_damas,
    }


def candidatos(engine, board: chess.Board, k: int, limite) -> list:
    """Top-k de Stockfish: lista de (move, eval_tras en cp desde el punto de vista de quien mueve)."""
    info = engine.analyse(board, limite, multipv=k)
    if isinstance(info, dict):
        info = [info]
    out = []
    for x in info:
        if "pv" in x and x["pv"]:
            out.append((x["pv"][0], score_cp(x["score"], board.turn)))
    return out
