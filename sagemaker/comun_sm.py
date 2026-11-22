"""
Lo que comparten los cinco scripts de SageMaker (EP3).

Solo usan boto3 y la biblioteca estandar, asi que corren en AWS CloudShell sin
instalar nada. Leen:
    modelo/metadatos.json   caso, tipo de modelo, columnas y clases
    modelo/xgboost/*.csv    los datos tratados en el formato del XGBoost

En AWS Academy Learner Lab no se pueden crear usuarios ni roles: todo usa el rol
LabRole, que ya viene en la cuenta.

Cada paso anota lo que hizo en sagemaker/estado.json. Si se corta la sesion de
CloudShell, vuelves a correr el mismo paso y retoma donde iba.
"""
import json
import sys
from pathlib import Path

import boto3

RAIZ = Path(__file__).resolve().parent.parent
MODELO_DIR = RAIZ / "modelo"
XGBOOST_DIR = MODELO_DIR / "xgboost"
ESTADO = RAIZ / "sagemaker" / "estado.json"

REGION = "us-east-1"
# Imagen oficial del algoritmo XGBoost de SageMaker en us-east-1, version 1.7-1
IMAGEN_XGBOOST = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"
TIPO_ENTRENAMIENTO = "ml.m5.large"

if not (MODELO_DIR / "metadatos.json").exists():
    sys.exit("Falta modelo/metadatos.json: corre antes python datos/preparar.py")
METADATOS = json.loads((MODELO_DIR / "metadatos.json").read_text(encoding="utf-8"))
CASO = METADATOS["caso"]

sesion = boto3.Session(region_name=REGION)
CUENTA = sesion.client("sts").get_caller_identity()["Account"]
ROL = f"arn:aws:iam::{CUENTA}:role/LabRole"
BUCKET = f"ity1102-ep3-{CUENTA}"
ENDPOINT = f"ep3-{CASO}"
PREFIJO = f"ep3-{CASO}-"

sm = sesion.client("sagemaker")
s3 = sesion.client("s3")


def estado():
    return json.loads(ESTADO.read_text()) if ESTADO.exists() else {}


def guardar_estado(**datos):
    actual = {**estado(), **datos}
    actual = {k: v for k, v in actual.items() if v is not None}
    ESTADO.write_text(json.dumps(actual, indent=2, ensure_ascii=False))
    return actual


def requiere(clave, paso):
    valor = estado().get(clave)
    if valor is None:
        sys.exit(f"Falta '{clave}'. Corre primero {paso}.")
    return valor
