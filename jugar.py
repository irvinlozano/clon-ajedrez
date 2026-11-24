"""Juega contra tu clon en el navegador.

Uso:
    python jugar.py            -> abre http://localhost:8000
    python jugar.py --temp 0   -> el clon siempre juega su jugada favorita (sin azar)
"""
import argparse
import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess

from clon import Clon

AQUI = Path(__file__).parent
LOCK = threading.Lock()
clon: Clon = None


def tablero_desde(q: dict) -> chess.Board:
    board = chess.Board()
    movs = q.get("moves", [""])[0]
    for u in filter(None, movs.split(",")):
        board.push_uci(u)
    return board


def estado(board: chess.Board) -> dict:
    piezas = {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()}
    legales = [{"uci": m.uci(), "san": board.san(m)} for m in board.legal_moves]
    res = None
    if board.is_game_over():
        o = board.outcome()
        res = {"ganador": None if o.winner is None else ("w" if o.winner else "b"),
               "motivo": o.termination.name.lower()}
    tmp = chess.Board()
    sans = []
    for m in board.move_stack:
        sans.append(tmp.san(m))
        tmp.push(m)
    ultima = board.move_stack[-1].uci() if board.move_stack else None
    return {"piezas": piezas, "turno": "w" if board.turn else "b", "legales": legales,
            "jaque": board.is_check(), "fin": res, "sans": sans, "ultima": ultima}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        self._enviar(200, "application/json", json.dumps(obj).encode())

    def _enviar(self, codigo, tipo, cuerpo: bytes):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._enviar(200, "text/html; charset=utf-8", (AQUI / "web" / "index.html").read_bytes())
        elif url.path == "/estado":
            # ?moves=e2e4,e7e5  -> tablero, jugadas legales y si terminó la partida
            board = tablero_desde(parse_qs(url.query))
            self._json(estado(board))
        elif url.path == "/jugada":
            board = tablero_desde(parse_qs(url.query))
            if board.is_game_over():
                self._json({"fin": True})
                return
            with LOCK:
                r = clon.elegir(board)
            cuerpo = {"uci": r["move"].uci(), "san": board.san(r["move"]), "origen": r["origen"],
                      "opciones": [{"uci": u, "san": board.san(chess.Move.from_uci(u)), "p": round(p, 3)}
                                   for u, p in r["opciones"][:5]]}
            self._json(cuerpo)
        else:
            self._enviar(404, "text/plain", b"no encontrado")


class ServidorLocal6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def main():
    global clon
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--sin-navegador", action="store_true", help="no abrir el navegador automáticamente")
    args = ap.parse_args()
    clon = Clon(temperatura=args.temp)
    # Solo en esta computadora: 127.0.0.1 (IPv4) y ::1 (IPv6), porque "localhost" puede resolver a cualquiera.
    srv = ThreadingHTTPServer(("127.0.0.1", args.puerto), Handler)
    try:
        srv6 = ServidorLocal6(("::1", args.puerto), Handler)
        threading.Thread(target=srv6.serve_forever, daemon=True).start()
    except OSError:
        pass
    url = f"http://localhost:{args.puerto}"
    print(f"Clon listo en {url}  (Ctrl+C para salir)")
    if not args.sin_navegador:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        clon.cerrar()


if __name__ == "__main__":
    main()
