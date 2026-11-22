"""
Paso 4: invoca el endpoint, mide el arranque en frio y evalua el modelo.

    python3 sagemaker/4_probar.py

El conjunto de prueba nunca subio a S3: el modelo lo ve por primera vez aqui.
Estas metricas son las de los apartados 1 y 5 del informe.

Para medir el arranque en frio de verdad, correlo despues de 10 minutos sin
usar el endpoint: SageMaker apaga el contenedor y la primera invocacion tiene
que encenderlo de nuevo (IE6).
"""
import csv
import re
import statistics
import time

from comun_sm import ENDPOINT, METADATOS, XGBOOST_DIR, guardar_estado, requiere, sesion

NUMERO = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
runtime = sesion.client("sagemaker-runtime")
CLASES = METADATOS["clases"]


def invocar(filas):
    """Devuelve (probabilidades por fila, milisegundos).

    Lee todos los numeros de la respuesta y los reparte por fila. El contenedor
    devuelve las probabilidades como "0.1,0.7,0.2" o como "[0.1, 0.7, 0.2]"
    segun la version, y asi funciona con las dos. Es lo mismo que hace
    app/motores.py en el servicio.
    """
    t = time.perf_counter()
    r = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv",
                                Accept="text/csv", Body="\n".join(",".join(f) for f in filas))
    ms = (time.perf_counter() - t) * 1000
    numeros = [float(x) for x in NUMERO.findall(r["Body"].read().decode())]
    k = len(CLASES)
    if len(numeros) != k * len(filas):
        raise SystemExit(f"El endpoint devolvio {len(numeros)} numeros; se esperaban "
                         f"{k} por cada una de las {len(filas)} filas")
    return [numeros[i:i + k] for i in range(0, len(numeros), k)], ms


def main():
    requiere("endpoint", "3_endpoint.py")
    with open(XGBOOST_DIR / "test.csv") as f:
        filas = list(csv.reader(f))[1:]
    reales = [f[0] for f in filas]
    x = [f[1:] for f in filas]

    # 1. La misma fila dos veces: la primera puede traer arranque en frio
    _, ms1 = invocar(x[:1])
    _, ms2 = invocar(x[:1])
    print(f"Endpoint {ENDPOINT}")
    print(f"  1.a invocacion: {ms1:7.0f} ms")
    print(f"  2.a invocacion: {ms2:7.0f} ms")

    # 2. Latencia en caliente: 20 invocaciones de una fila
    tiempos = sorted(invocar(x[:1])[1] for _ in range(20))
    p50, p95 = statistics.median(tiempos), tiempos[18]
    print(f"  20 en caliente: p50 {p50:.0f} ms · p95 {p95:.0f} ms "
          f"(medido desde donde corres el script, incluye el viaje a Virginia)")

    # 3. Metricas sobre la prueba, en lotes de 100 filas por invocacion
    probas = []
    for i in range(0, len(x), 100):
        probas += invocar(x[i:i + 100])[0]

    pred = [CLASES[max(range(len(f)), key=f.__getitem__)] for f in probas]
    aciertos = sum(1 for a, b in zip(reales, pred) if a == b)
    f1s = []
    for c in CLASES:
        vp = sum(1 for a, b in zip(reales, pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(reales, pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(reales, pred) if a == c and b != c)
        pr = vp / (vp + fp) if vp + fp else 0.0
        rc = vp / (vp + fn) if vp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    mas_frecuente = max(set(reales), key=reales.count)

    resultado = {
        "cold_ms": round(ms1), "caliente_ms": round(ms2),
        "p50_ms": round(p50), "p95_ms": round(p95),
        "filas_prueba": len(reales),
        "accuracy": round(aciertos / len(reales), 4),
        "f1_macro": round(sum(f1s) / len(f1s), 4),
        "f1_por_clase": dict(zip(CLASES, [round(v, 3) for v in f1s])),
        "linea_base_clase_mas_frecuente": round(reales.count(mas_frecuente) / len(reales), 4),
    }
    print(f"Conjunto de prueba: {len(reales)} filas")
    for k, v in resultado.items():
        print(f"  {k}: {v}")
    print("Comparalo con modelo/METRICAS.md (el modelo local): deberian parecerse. "
          "Si no, el CSV o el orden de las columnas no es el mismo.")
    guardar_estado(prueba=resultado)


if __name__ == "__main__":
    main()
