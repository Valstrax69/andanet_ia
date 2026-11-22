"""
Paso 2: entrena el modelo con un Training Job de SageMaker (XGBoost integrado).

    python3 sagemaker/2_entrenar.py

SageMaker enciende una ml.m5.large, entrena con los CSV de S3, guarda el modelo
en S3 y apaga la maquina. Se cobra solo el tiempo encendida (2 a 5 minutos).

Si cortas el script el entrenamiento sigue: vuelves a correrlo y retoma ese
trabajo en vez de lanzar otro.

Son seis intenciones, asi que el objetivo es multi:softprob: devuelve una
probabilidad por clase, igual que el predict_proba del modelo local de EP2.
"""
import time
from datetime import datetime

from comun_sm import (BUCKET, CASO, IMAGEN_XGBOOST, METADATOS, PREFIJO, ROL,
                      TIPO_ENTRENAMIENTO, estado, guardar_estado, requiere, sm)


def hiperparametros():
    # DECISION a verificar (DESAFIO S2): estos valores son un punto de partida
    # razonable, no optimos. Si los cambias, compara la metrica de 4_probar.py
    # antes y despues y deja las dos cifras en el informe.
    return {
        "num_round": "150",
        "max_depth": "5",
        "eta": "0.1",
        "subsample": "0.8",
        "min_child_weight": "3",
        "objective": "multi:softprob",
        "num_class": str(len(METADATOS["clases"])),
        "eval_metric": "mlogloss",
    }


def canal(nombre):
    return {
        "ChannelName": nombre,
        "ContentType": "text/csv",
        "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix",
            "S3Uri": f"s3://{BUCKET}/{CASO}/{nombre}/",
            "S3DataDistributionType": "FullyReplicated"}},
    }


def main():
    requiere("datos", "1_subir.py")

    previo = estado().get("trabajo_en_curso")
    if previo and sm.describe_training_job(
            TrainingJobName=previo)["TrainingJobStatus"] in ("InProgress", "Completed"):
        trabajo = previo
        print(f"Retomo el Training Job {trabajo}.")
    else:
        trabajo = f"{PREFIJO}{datetime.now():%Y%m%d-%H%M%S}"
        hp = hiperparametros()
        sm.create_training_job(
            TrainingJobName=trabajo,
            AlgorithmSpecification={"TrainingImage": IMAGEN_XGBOOST, "TrainingInputMode": "File"},
            RoleArn=ROL,
            InputDataConfig=[canal("train"), canal("validation")],
            OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{CASO}/modelos/"},
            ResourceConfig={"InstanceType": TIPO_ENTRENAMIENTO, "InstanceCount": 1,
                            "VolumeSizeInGB": 5},
            StoppingCondition={"MaxRuntimeInSeconds": 1200},
            HyperParameters=hp,
        )
        guardar_estado(trabajo_en_curso=trabajo, hiperparametros=hp)
        print(f"Training Job {trabajo} en {TIPO_ENTRENAMIENTO} · multi:softprob con "
              f"{hp['num_class']} clases. Se revisa cada 20 s.")

    inicio, anterior = time.time(), None
    while True:
        d = sm.describe_training_job(TrainingJobName=trabajo)
        actual = (d["TrainingJobStatus"], d.get("SecondaryStatus"))
        if actual != anterior:
            print(f"  {int(time.time() - inicio):>4} s · {actual[0]} · {actual[1]}")
            anterior = actual
        if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(20)

    if d["TrainingJobStatus"] != "Completed":
        guardar_estado(trabajo_en_curso=None)
        raise SystemExit(f"El entrenamiento termino en {d['TrainingJobStatus']}: "
                         f"{d.get('FailureReason')}")

    for m in d.get("FinalMetricDataList", []):
        print(f"  {m['MetricName']}: {m['Value']:.4f}")
    artefacto = d["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"  Tiempo facturado: {d.get('BillableTimeInSeconds', 0)} s")
    print(f"  Modelo: {artefacto}")
    print("Siguiente: python3 sagemaker/3_endpoint.py")
    guardar_estado(trabajo=trabajo, trabajo_en_curso=None, artefacto=artefacto,
                   segundos_facturados=d.get("BillableTimeInSeconds", 0))


if __name__ == "__main__":
    main()
