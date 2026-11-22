"""
Paso 3: publica el modelo en un endpoint serverless, o lo actualiza.

    python3 sagemaker/3_endpoint.py                           # 2048 MB, concurrencia 10
    python3 sagemaker/3_endpoint.py --memoria 3072 --concurrencia 5
    python3 sagemaker/3_endpoint.py --tiempo-real             # plan B: COBRA POR HORA

El nombre del endpoint (ep3-chatbot) no cambia nunca: el microservicio de IA lo
invoca por ese nombre y no se entera de que el modelo cambio. La version que ve
es el nombre de la configuracion nueva.

DECISION a verificar (DESAFIO S3) · memoria y concurrencia. La concurrencia
decide cuantas invocaciones simultaneas acepta el endpoint; lo que pasa de ahi
lo rechaza con ThrottlingException y el dominio tiene que derivar. En las
mediciones del docente con concurrencia 2 un 23% de las solicitudes salieron por
el camino de reserva, y con 10 solo el 1%. Por eso el valor por omision aqui es
10, pero hay que medirlo en nuestra cuenta antes de defenderlo (IE6).
"""
import argparse
import time
from datetime import datetime

from botocore.exceptions import ClientError

from comun_sm import ENDPOINT, IMAGEN_XGBOOST, PREFIJO, ROL, guardar_estado, requiere, sm

EN_TRANSITO = ("Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting")


def estado_endpoint():
    try:
        return sm.describe_endpoint(EndpointName=ENDPOINT)["EndpointStatus"]
    except ClientError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memoria", type=int, default=2048, help="MB del endpoint serverless (1024 a 6144)")
    p.add_argument("--concurrencia", type=int, default=10, help="invocaciones simultaneas")
    p.add_argument("--tiempo-real", action="store_true", help="plan B si el lab rechaza serverless")
    args = p.parse_args()
    artefacto = requiere("artefacto", "2_entrenar.py")

    estado = estado_endpoint()
    while estado in EN_TRANSITO:
        print(f"  El endpoint esta en {estado}: espero (cada 15 s).")
        time.sleep(15)
        estado = estado_endpoint()

    sello = f"{datetime.now():%Y%m%d-%H%M%S}"
    modelo = config = f"{PREFIJO}{sello}"
    sm.create_model(ModelName=modelo, ExecutionRoleArn=ROL,
                    PrimaryContainer={"Image": IMAGEN_XGBOOST, "ModelDataUrl": artefacto})

    variante = {"VariantName": "principal", "ModelName": modelo}
    if args.tiempo_real:
        variante.update(InstanceType="ml.t2.medium", InitialInstanceCount=1)
        tipo = "tiempo real · ml.t2.medium · COBRA CADA HORA: borralo con 5_borrar.py al terminar"
    else:
        variante["ServerlessConfig"] = {"MemorySizeInMB": args.memoria,
                                        "MaxConcurrency": args.concurrencia}
        tipo = f"serverless · {args.memoria} MB · concurrencia {args.concurrencia}"
    sm.create_endpoint_config(EndpointConfigName=config, ProductionVariants=[variante])

    if estado == "Failed":
        sm.delete_endpoint(EndpointName=ENDPOINT)
        sm.get_waiter("endpoint_deleted").wait(EndpointName=ENDPOINT)
        estado = None
    if estado is None:
        sm.create_endpoint(EndpointName=ENDPOINT, EndpointConfigName=config)
        print(f"Creando {ENDPOINT} ({tipo})")
    else:
        sm.update_endpoint(EndpointName=ENDPOINT, EndpointConfigName=config)
        print(f"Actualizando {ENDPOINT} a la configuracion {config} ({tipo})")

    inicio, anterior = time.time(), None
    while True:
        d = sm.describe_endpoint(EndpointName=ENDPOINT)
        if d["EndpointStatus"] != anterior:
            print(f"  {int(time.time() - inicio):>4} s · {d['EndpointStatus']}")
            anterior = d["EndpointStatus"]
        if d["EndpointStatus"] in ("InService", "Failed"):
            break
        time.sleep(15)

    if d["EndpointStatus"] != "InService":
        raise SystemExit(f"El endpoint quedo en Failed: {d.get('FailureReason')}"
                         f"  ·  Plan B: --tiempo-real")
    print(f"Endpoint {ENDPOINT} InService · version {config}.")
    print("Siguiente: python3 sagemaker/4_probar.py   ·   En la EC2: docker compose restart ia")
    guardar_estado(endpoint=ENDPOINT, modelo=modelo, version=config, tipo=tipo,
                   memoria=args.memoria, concurrencia=args.concurrencia)


if __name__ == "__main__":
    main()
