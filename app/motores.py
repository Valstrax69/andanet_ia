"""
Donde vive el modelo. El resto del servicio no sabe cual de los dos esta usando.

    MOTOR=local      EP2: el clasificador viene dentro de la imagen (modelo.pkl)
    MOTOR=sagemaker  EP3: el clasificador vive en un endpoint de SageMaker

Los dos reciben filas ya convertidas a numeros por app/caracteristicas.py y
devuelven, por fila, la probabilidad de cada clase en el orden de
metadatos["clases"].

La vectorizacion NO cambia entre motores: el TF-IDF sigue corriendo aqui en el
servicio. Lo unico que se mueve a AWS es el clasificador. Por eso el contrato de
/predict es el mismo en los dos casos.
"""
import json
import os
import re
import time
from pathlib import Path

MODELO_PATH = Path(os.getenv("MODELO_PATH", "/app/modelo/modelo.pkl"))
META_PATH = Path(os.getenv("META_PATH", "/app/modelo/metadatos.json"))

# El contenedor devuelve las probabilidades como "0.1,0.7,0.2" o como
# "[0.1, 0.7, 0.2]" segun la version. Se leen todos los numeros y se reparten.
NUMERO = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def cargar_metadatos():
    return json.loads(META_PATH.read_text(encoding="utf-8"))


class MotorLocal:
    nombre = "local"

    def __init__(self, metadatos):
        import joblib
        self.modelo = joblib.load(MODELO_PATH)
        self.clases = [str(c) for c in self.modelo.classes_]
        self._version = metadatos.get("modelo_version", "desconocida")

    @property
    def version(self):
        return self._version

    def probabilidades(self, filas):
        return self.modelo.predict_proba(filas).tolist()

    def estado(self):
        return "cargado"


class MotorSageMaker:
    """Invoca el endpoint con el rol de la EC2: no hay claves de AWS en el codigo."""
    nombre = "sagemaker"

    def __init__(self, metadatos):
        import boto3
        from botocore.config import Config

        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.endpoint = os.getenv("ENDPOINT") or f"ep3-{metadatos['caso']}"
        self.clases = metadatos["clases"]

        # Sin reintentos y con tiempo maximo: si SageMaker no responde hay que
        # saberlo rapido para que el dominio derive a un ejecutivo, en vez de
        # dejar al cliente esperando. 10 s cubren un arranque en frio.
        config = Config(connect_timeout=2,
                        read_timeout=float(os.getenv("SAGEMAKER_TIMEOUT", "10")),
                        retries={"total_max_attempts": 1, "mode": "standard"})
        self.runtime = boto3.client("sagemaker-runtime", region_name=region, config=config)
        self.sagemaker = boto3.client("sagemaker", region_name=region, config=config)
        self._version = None
        self._version_hasta = 0.0

    @property
    def version(self):
        """La version es el nombre de la configuracion del endpoint.

        Cambia cada vez que se publica un modelo nuevo. Se le pregunta a
        SageMaker una vez por minuto como maximo, no en cada prediccion.
        """
        if time.monotonic() > self._version_hasta:
            try:
                d = self.sagemaker.describe_endpoint(EndpointName=self.endpoint)
                self._version = d["EndpointConfigName"]
                self._version_hasta = time.monotonic() + 60
            except Exception:
                return self._version or "desconocida"
        return self._version

    def probabilidades(self, filas):
        cuerpo = "\n".join(",".join(repr(float(v)) for v in fila) for fila in filas)
        r = self.runtime.invoke_endpoint(EndpointName=self.endpoint,
                                         ContentType="text/csv", Accept="text/csv",
                                         Body=cuerpo)
        numeros = [float(x) for x in NUMERO.findall(r["Body"].read().decode())]
        k = len(self.clases)
        if len(numeros) != k * len(filas):
            raise ValueError(f"SageMaker devolvio {len(numeros)} numeros; se esperaban "
                             f"{k} por cada una de las {len(filas)} filas")
        return [numeros[i:i + k] for i in range(0, len(numeros), k)]

    def estado(self):
        try:
            return self.sagemaker.describe_endpoint(EndpointName=self.endpoint)["EndpointStatus"]
        except Exception as e:
            return f"error: {type(e).__name__}"


def crear_motor(metadatos):
    motor = os.getenv("MOTOR", "local")
    if motor not in ("local", "sagemaker"):
        raise SystemExit(f"MOTOR={motor} no existe. Usa 'local' o 'sagemaker'.")
    return {"local": MotorLocal, "sagemaker": MotorSageMaker}[motor](metadatos)
