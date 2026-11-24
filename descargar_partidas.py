"""Descarga las partidas de un usuario de chess.com (API pública) y guarda solo las de un ritmo.

Uso:
    python descargar_partidas.py irvinlozano rapid
Salida:
    datos/<usuario>_<ritmo>.pgn  y  datos/<usuario>_<ritmo>.json (metadatos)
"""
import json
import sys
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "clon-ajedrez-proyecto-escolar/1.0"}


def main(usuario: str, ritmo: str = "rapid") -> None:
    usuario = usuario.lower()
    out = Path(__file__).parent / "datos"
    out.mkdir(exist_ok=True)

    r = requests.get(f"https://api.chess.com/pub/player/{usuario}/games/archives", headers=HEADERS, timeout=30)
    r.raise_for_status()
    archivos = r.json()["archives"]
    print(f"{len(archivos)} meses con partidas")

    partidas = []
    for url in archivos:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        for g in resp.json().get("games", []):
            if g.get("time_class") == ritmo and g.get("rules") == "chess" and g.get("pgn"):
                partidas.append(g)
        time.sleep(0.3)  # no saturar la API

    (out / f"{usuario}_{ritmo}.pgn").write_text("\n\n".join(g["pgn"] for g in partidas), encoding="utf-8")
    meta = [
        {
            "url": g.get("url"),
            "fin": g.get("end_time"),
            "ritmo": g.get("time_control"),
            "blancas": g["white"]["username"], "elo_blancas": g["white"].get("rating"), "res_blancas": g["white"].get("result"),
            "negras": g["black"]["username"], "elo_negras": g["black"].get("rating"), "res_negras": g["black"].get("result"),
        }
        for g in partidas
    ]
    (out / f"{usuario}_{ritmo}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{len(partidas)} partidas de {ritmo} guardadas en {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rapid")
