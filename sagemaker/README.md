# sagemaker/ · el modelo en un endpoint serverless (EP3)

Los cinco pasos corren en **AWS CloudShell** (el icono `>_` de la consola,
region `us-east-1`). Solo usan `boto3`, que CloudShell ya trae: no hay que
instalar nada.

```bash
git clone https://github.com/Valstrax69/andanet_ia.git
cd andanet_ia
python3 sagemaker/1_subir.py      # sube train y validation a S3
python3 sagemaker/2_entrenar.py   # Training Job XGBoost (2 a 5 min)
python3 sagemaker/3_endpoint.py   # endpoint serverless ep3-chatbot (unos 3 min)
python3 sagemaker/4_probar.py     # arranque en frio, latencia y metricas
```

No hace falta correr `datos/preparar.py` en CloudShell: los CSV del XGBoost
estan versionados en `modelo/xgboost/`. CloudShell trae Python 3.9 y nuestro
`requirements-dev.txt` pide numpy 2.1.3, que necesita 3.10 o mas, asi que
instalarlo ahi falla.

Cada paso anota en `sagemaker/estado.json` lo que hizo. Si se corta la sesion,
se vuelve a correr el mismo paso y retoma.

Al terminar la presentacion: `python3 sagemaker/5_borrar.py`.

## Lo que queda en AWS

| Paso | Que crea | Donde verlo en la consola |
|---|---|---|
| 1 | Bucket `ity1102-ep3-<cuenta>` con train y validation | S3 |
| 2 | Training Job `ep3-chatbot-<fecha>` y el modelo en S3 | SageMaker → Training jobs |
| 3 | Modelo, configuracion y endpoint `ep3-chatbot` | SageMaker → Inference → Endpoints |
| 4 | Nada: solo invoca y mide | CloudWatch → metricas del endpoint |

## Conectarlo al sistema

En el `.env` de la EC2: `MOTOR=sagemaker` y `ENDPOINT=ep3-chatbot`. Despues
`docker compose up -d ia` y `curl -s localhost:8080/api/health` tiene que decir
`"motor": "sagemaker"`.
