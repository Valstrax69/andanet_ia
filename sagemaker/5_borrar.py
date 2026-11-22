"""
Paso 5: borra el endpoint, sus configuraciones y sus modelos.

    python3 sagemaker/5_borrar.py           # despues de la presentacion
    python3 sagemaker/5_borrar.py --todo    # tambien el bucket

Un endpoint serverless sin uso no cobra; uno de tiempo real cobra cada hora, asi
que si usaste --tiempo-real hay que borrarlo al terminar la sesion.

El modelo entrenado queda en S3 (salvo con --todo): volver a publicar el
endpoint con 3_endpoint.py toma minutos y no hay que reentrenar.
"""
import sys

from botocore.exceptions import ClientError

from comun_sm import BUCKET, ENDPOINT, ESTADO, PREFIJO, guardar_estado, sesion, sm


def main():
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT)
        print(f"Endpoint borrado: {ENDPOINT}")
    except ClientError:
        print(f"No habia endpoint {ENDPOINT}")

    for c in sm.list_endpoint_configs(NameContains=PREFIJO, MaxResults=100)["EndpointConfigs"]:
        sm.delete_endpoint_config(EndpointConfigName=c["EndpointConfigName"])
        print(f"Configuracion borrada: {c['EndpointConfigName']}")

    for m in sm.list_models(NameContains=PREFIJO, MaxResults=100)["Models"]:
        sm.delete_model(ModelName=m["ModelName"])
        print(f"Modelo borrado: {m['ModelName']}")

    if "--todo" in sys.argv:
        try:
            sesion.resource("s3").Bucket(BUCKET).objects.all().delete()
            sesion.client("s3").delete_bucket(Bucket=BUCKET)
            print(f"Bucket borrado: {BUCKET}")
        except ClientError as e:
            print(f"No se pudo borrar el bucket {BUCKET}: {e}")
        ESTADO.unlink(missing_ok=True)
    else:
        guardar_estado(endpoint=None, modelo=None, version=None, tipo=None)


if __name__ == "__main__":
    main()
