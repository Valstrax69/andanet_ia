"""
Microservicio de IA - clasificador de intencion.

Carga el modelo entrenado por datos/preparar.py y expone:
  GET  /health   -> estado y version del modelo
  POST /predict  -> {"texto": "..."} -> intencion, confianza, modelo_version

Antes de clasificar enmascara el texto (RNF5): el modelo nunca ve datos personales.
"""
import json
import os
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.anonimizar import enmascarar

MODELO_PATH = Path(os.getenv("MODELO_PATH", "/app/modelo/modelo.pkl"))
META_PATH = Path(os.getenv("META_PATH", "/app/modelo/modelo_meta.json"))

modelo = joblib.load(MODELO_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
MODELO_VERSION = meta.get("modelo_version", "desconocida")

app = FastAPI(title="Andesnet IA - clasificador de intencion")


class Peticion(BaseModel):
    texto: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok", "modelo_version": MODELO_VERSION}


@app.post("/predict")
def predict(p: Peticion):
    texto_limpio = enmascarar(p.texto)
    if not texto_limpio:
        return {"intencion": "otra", "confianza": 0.0,
                "modelo_version": MODELO_VERSION, "entrada_enmascarada": ""}
    intencion = modelo.predict([texto_limpio])[0]
    proba = modelo.predict_proba([texto_limpio])[0]
    confianza = float(max(proba))
    return {
        "intencion": intencion,
        "confianza": round(confianza, 4),
        "modelo_version": MODELO_VERSION,
        "entrada_enmascarada": texto_limpio,
    }
