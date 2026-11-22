"""
Variables del clasificador de intencion: el texto se convierte en numeros.

El modelo no lee palabras, lee numeros. Quien hace la conversion es el
vectorizador TF-IDF que entreno datos/preparar.py y que queda guardado en
modelo/vectorizador.pkl.

TF-IDF cuenta cada palabra y cada par de palabras seguidas (bigrama), y despues
le baja el peso a las que aparecen en muchos mensajes. Esa es la parte que
importa: "mi" y "quiero" salen en todos los mensajes y no distinguen nada, en
cambio "boleta" o "router" si. Son 897 columnas, fijas, en el orden que guarda
metadatos.json.

Este modulo lo usan los dos motores. El de EP2 vectoriza y predice en el mismo
proceso; el de EP3 vectoriza aqui y manda los 897 numeros al endpoint de
SageMaker. Asi el entrenamiento y los dos motores ven exactamente lo mismo.
"""
import html
import os
import re
from pathlib import Path

import joblib

from app.anonimizar import enmascarar

VECTORIZADOR_PATH = Path(os.getenv("VECTORIZADOR_PATH", "/app/modelo/vectorizador.pkl"))

RE_ESPACIOS = re.compile(r"\s+")

_vectorizador = None


def sin_entidades(texto):
    """Arregla las entidades HTML del archivo. Va ANTES de enmascarar, porque un
    correo escrito con "&" no lo reconoce el enmascarador.

    En estos datos "&amp;" reemplaza a la letra "y" (se ve "a&amp;uda" por
    "ayuda" y "@&amp;ahoo.es" por "@yahoo.es"), asi que se devuelve la "y".
    """
    return html.unescape(str(texto or "")).replace("&", "y")


def limpiar(texto):
    """Texto legible y sin datos personales.

    Es lo unico que ve el modelo, lo que se guarda en la base y lo que se
    devuelve al dominio. La misma funcion se usa al entrenar y al predecir.
    """
    return RE_ESPACIOS.sub(" ", enmascarar(sin_entidades(texto))).strip()


def cargar_vectorizador(path=None):
    """Carga el vectorizador una sola vez y lo deja en memoria."""
    global _vectorizador
    if _vectorizador is None:
        _vectorizador = joblib.load(path or VECTORIZADOR_PATH)
    return _vectorizador


def vectorizar(texto, vectorizador=None):
    """Devuelve la lista de numeros del texto, en el orden de las columnas.

    preparar.py pasa su propio vectorizador recien entrenado; el servicio no
    pasa nada y se usa el que esta guardado en la imagen.
    """
    v = vectorizador or cargar_vectorizador()
    return v.transform([limpiar(texto)]).toarray()[0].tolist()


def columnas(vectorizador=None):
    v = vectorizador or cargar_vectorizador()
    return list(v.get_feature_names_out())
