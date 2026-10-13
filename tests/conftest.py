import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODELO_PATH", str(RAIZ / "modelo" / "modelo.pkl"))
os.environ.setdefault("META_PATH", str(RAIZ / "modelo" / "modelo_meta.json"))
