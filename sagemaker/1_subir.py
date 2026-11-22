"""
Paso 1: sube a S3 los CSV de entrenamiento y validacion.

    python3 sagemaker/1_subir.py

La prueba (test.csv) NO sube: queda local para que el modelo la vea por primera
vez en 4_probar.py. Si subiera, el numero del informe no mediria nada.
"""
from botocore.exceptions import ClientError

from comun_sm import BUCKET, CASO, METADATOS, XGBOOST_DIR, guardar_estado, s3


def main():
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)   # en us-east-1 no lleva LocationConstraint
        print(f"Bucket creado: {BUCKET}")

    for nombre in ("train", "validation"):
        archivo = XGBOOST_DIR / f"{nombre}.csv"
        s3.upload_file(str(archivo), BUCKET, f"{CASO}/{nombre}/{nombre}.csv")
        filas = sum(1 for _ in open(archivo))
        print(f"  s3://{BUCKET}/{CASO}/{nombre}/{nombre}.csv  ·  {filas} filas")

    print(f"Caso {CASO} · {len(METADATOS['columnas'])} variables · "
          f"clases: {', '.join(METADATOS['clases'])}")
    print("Siguiente: python3 sagemaker/2_entrenar.py")
    guardar_estado(caso=CASO, datos=f"s3://{BUCKET}/{CASO}/")


if __name__ == "__main__":
    main()
