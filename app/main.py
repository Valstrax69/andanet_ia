"""
Microservicio de IA - clasificador de intencion.

Expone:
  GET  /health   -> estado del motor y version del modelo
  POST /predict  -> {"texto": "..."} -> intencion, confianza, modelo_version

El texto se enmascara antes de cualquier cosa (RNF5): el modelo nunca ve datos
personales, y lo que se devuelve al dominio para guardar tampoco los tiene.

Son dos pasos separados:
  1. app.caracteristicas vectoriza el texto con el TF-IDF (826 columnas)
  2. app.motores entrega esos numeros al clasificador

El paso 1 no cambia nunca. El paso 2 es el que decide MOTOR: en EP2 el modelo
esta dentro de la imagen, en EP3 vive en un endpoint de SageMaker. El servicio
responde igual en los dos casos.
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.caracteristicas import cargar_vectorizador, limpiar, vectorizar
from app.motores import cargar_metadatos, crear_motor

META = cargar_metadatos()
MODELO_VERSION = META.get("modelo_version", "desconocida")
CLASES = META.get("clases", [])

cargar_vectorizador()          # al arrancar, no en la primera peticion
motor = crear_motor(META)

app = FastAPI(title="Andesnet IA - clasificador de intencion",
              version=MODELO_VERSION)


class Peticion(BaseModel):
    texto: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "motor": motor.nombre,
        "motor_estado": motor.estado(),
        "modelo_version": motor.version,
        "modelo_datos_version": MODELO_VERSION,
        "clases": len(CLASES),
        "endpoint": os.getenv("ENDPOINT", "") if motor.nombre == "sagemaker" else "",
    }


@app.post("/predict")
def predict(p: Peticion):
    texto_limpio = limpiar(p.texto)
    if not texto_limpio:
        return {"intencion": "otra", "confianza": 0.0,
                "modelo_version": motor.version, "entrada_enmascarada": "",
                "motor": motor.nombre}

    proba = motor.probabilidades([vectorizar(p.texto)])[0]
    i = max(range(len(proba)), key=proba.__getitem__)
    return {
        "intencion": motor.clases[i],
        "confianza": round(float(proba[i]), 4),
        "modelo_version": motor.version,
        "entrada_enmascarada": texto_limpio,
        "motor": motor.nombre,
    }
