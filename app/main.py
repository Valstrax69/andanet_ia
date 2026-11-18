"""
Microservicio de IA - clasificador de intencion.

Expone:
  GET  /health   -> estado y version del modelo
  POST /predict  -> {"texto": "..."} -> intencion, confianza, modelo_version

El texto se enmascara antes de cualquier cosa (RNF5): el modelo nunca ve datos
personales, y lo que se devuelve al dominio para guardar tampoco los tiene.

Son dos pasos separados a proposito:
  1. app.caracteristicas vectoriza el texto (TF-IDF, 897 columnas)
  2. el clasificador recibe esos numeros

En EP3 el paso 2 se reemplaza por una llamada al endpoint de SageMaker, y el
paso 1 no cambia: por eso el vectorizador vive aparte del clasificador.
"""
import json
import os
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.caracteristicas import cargar_vectorizador, limpiar, vectorizar

MODELO_PATH = Path(os.getenv("MODELO_PATH", "/app/modelo/modelo.pkl"))
META_PATH = Path(os.getenv("META_PATH", "/app/modelo/metadatos.json"))

modelo = joblib.load(MODELO_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
MODELO_VERSION = meta.get("modelo_version", "desconocida")
CLASES = meta.get("clases", [])

cargar_vectorizador()   # al arrancar, no en la primera peticion

app = FastAPI(title="Andesnet IA - clasificador de intencion")


class Peticion(BaseModel):
    texto: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok", "modelo_version": MODELO_VERSION,
            "motor": "local", "clases": len(CLASES)}


@app.post("/predict")
def predict(p: Peticion):
    texto_limpio = limpiar(p.texto)
    if not texto_limpio:
        return {"intencion": "otra", "confianza": 0.0,
                "modelo_version": MODELO_VERSION, "entrada_enmascarada": ""}

    proba = modelo.predict_proba([vectorizar(p.texto)])[0]
    i = int(proba.argmax())
    return {
        "intencion": str(modelo.classes_[i]),
        "confianza": round(float(proba[i]), 4),
        "modelo_version": MODELO_VERSION,
        "entrada_enmascarada": texto_limpio,
    }
