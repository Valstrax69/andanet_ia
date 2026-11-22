"""
preparar.py - Tratamiento de datos y entrenamiento del modelo de intencion.

Lee los CSV crudos del Caso 2 (telco), los perfila, los limpia con reglas
documentadas, enmascara datos personales, arma el dataset, entrena el
clasificador y guarda:
  - datos/tratados/{clientes,faq,tickets}.csv  -> los usa el servicio de carga
  - modelo/vectorizador.pkl                    -> el TF-IDF entrenado
  - modelo/modelo.pkl                          -> el clasificador
  - modelo/metadatos.json                      -> columnas, clases y metricas
  - modelo/xgboost/{train,validation,test}.csv -> los usa SageMaker en EP3
  - modelo/METRICAS.md                         -> la tabla para el informe

Si el macro-F1 queda bajo el umbral, termina con codigo 1 para que el pipeline
no construya la imagen.

Tres decisiones medidas sobre estos datos, distintas de la version de EP2:

1. El dataset son SOLO los tickets. Las conversaciones traen la etiqueta que
   escribio el bot de reglas anterior: 270 textos aparecen en las dos fuentes y
   166 de ellos llevan etiqueta distinta segun la fuente. Entrenar con las dos
   le ensena al modelo a repetir los errores del bot viejo (macro-F1 0,699 con
   conversaciones contra 0,931 sin ellas).

2. La particion es por plantilla, no por fila. Los tickets vienen de plantillas:
   1.489 textos unicos son 752 plantillas distintas, asi que partiendo por fila
   el 58% de los textos de prueba eran una plantilla ya vista. Partiendo por
   plantilla ninguna cae en los dos lados, y el resultado es practicamente el
   mismo: el modelo aprende el vocabulario de la intencion, no la plantilla.

3. Se mantiene TF-IDF y no feature hashing, que era el otro camino para generar
   el CSV numerico que pide el XGBoost de SageMaker. Medido con 18 frases
   escritas a mano, fuera de las plantillas: TF-IDF acierta 15 y el hashing 10.
   Con el umbral de confianza en 0,55, TF-IDF responde 6 y acierta las 6,
   mientras el hashing responde 10 y acierta 6. Le falta el idf: contando nomas,
   "quiero" y "mi" pesan igual que "boleta". Y el vocabulario cabe en 897
   columnas fijas, que era la razon por la que se evaluo el hashing.

Uso:
    python datos/preparar.py            # usa rutas por defecto
    UMBRAL_F1=0.70 python datos/preparar.py
"""
import os
import re
import sys
import json
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- Rutas ---
BASE = Path(__file__).resolve().parent          # .../ia/datos
sys.path.insert(0, str(BASE.parent))            # para importar app.*
from app.anonimizar import enmascarar, tiene_pii  # noqa: E402
from app.caracteristicas import limpiar  # noqa: E402

RAW = BASE / "raw"
TRATADOS = BASE / "tratados"
MODELO_DIR = BASE.parent / "modelo"             # .../ia/modelo
XGBOOST_DIR = MODELO_DIR / "xgboost"
TRATADOS.mkdir(parents=True, exist_ok=True)
MODELO_DIR.mkdir(parents=True, exist_ok=True)

CASO = "chatbot"
# 2.0.0: el modelo se parte en dos archivos (vectorizador y clasificador) para
# que EP3 pueda vectorizar aqui y predecir en SageMaker. El contrato del
# servicio no cambia, pero el artefacto si.
MODELO_VERSION = "2.0.0"
UMBRAL_F1 = float(os.getenv("UMBRAL_F1", "0.65"))
SEMILLA = 42

# Las seis intenciones del caso
INTENCIONES = ["facturacion", "falla_conexion", "cambio_plan", "visita", "anulacion", "otra"]


# --- Helpers de texto ---
def sin_acentos(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def norm(s):
    # minuscula, sin acentos, guiones_bajos como espacios
    return sin_acentos(str(s)).lower().replace("_", " ").strip()


def mapear_intencion(valor):
    t = norm(valor)
    if any(k in t for k in ["anula", "baja", "portab", "cancel", "fuga"]):
        return "anulacion"
    if any(k in t for k in ["visita", "agenda", "terreno"]):
        return "visita"
    if any(k in t for k in ["cambio plan", "upgrade", "downgrade", "plan"]):
        return "cambio_plan"
    if any(k in t for k in ["factur", "cobro", "boleta", "pago", "saldo", "fact"]):
        return "facturacion"
    if any(k in t for k in ["falla", "soporte", "tecnic", "cobertura", "internet",
                            "conexion", "senal", "router", "equipos", "caida"]):
        return "falla_conexion"
    return "otra"


RE_ENLACE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE | re.DOTALL)
RE_HTML = re.compile(r"<[^>]+>")
RE_DIGITOS = re.compile(r"\d+")

# La categoria de la FAQ no sirve para clasificar: el mismo texto aparece con
# categorias distintas (el articulo del router figura como "Facturacion"). La
# pregunta si corresponde con la respuesta, asi que la intencion sale de ahi.
PALABRAS_FAQ = [
    ("anulacion",      ["anula", "baja", "portab", "cancel"]),
    ("visita",         ["visita", "agend", "tecnico", "terreno", "instalacion"]),
    ("cambio_plan",    ["cambio de plan", "cambiar de plan", "cambio mi plan",
                        "cambiar mi plan", "cambio plan", "upgrade", "downgrade"]),
    ("facturacion",    ["factura", "boleta", "cobro", "saldo", "pagar", "pago atrasado",
                        "cuanto debo", "vence", "deuda", "como pago"]),
    ("falla_conexion", ["internet", "conexion", "senal", "router", "falla", "lento",
                        "no funciona", "corte", "wifi", "equipo"]),
]


def intencion_de_pregunta(texto):
    t = norm(texto)
    for intencion, palabras in PALABRAS_FAQ:
        if any(p in t for p in palabras):
            return intencion
    return "otra"


def limpiar_html(texto):
    # El gestor de contenidos viejo dejo etiquetas y enlaces "Ver mas" dentro
    # de la respuesta. El enlace se borra completo, con su texto.
    t = RE_ENLACE.sub(" ", str(texto))
    return re.sub(r"\s+", " ", RE_HTML.sub(" ", t)).strip()


def plantilla(texto):
    """La misma frase con otro numero es la misma plantilla.

    Los tickets se generaron desde plantillas: "boleta de 45000" y "boleta de
    78000" son el mismo texto para efectos de aprender. Se usa para que una
    plantilla no quede en entrenamiento y en prueba a la vez.
    """
    return RE_DIGITOS.sub("#", str(texto).lower()).strip()


# Mapa de regiones a nombre canonico
MAPA_REGION = {
    "i": "Tarapacá", "ii": "Antofagasta", "iii": "Atacama", "iv": "Coquimbo",
    "v": "Valparaíso", "vi": "O'Higgins", "vii": "Maule", "viii": "Biobío",
    "ix": "Araucanía", "x": "Los Lagos", "xiv": "Los Ríos", "xvi": "Ñuble",
    "rm": "Metropolitana", "region metropolitana": "Metropolitana",
    "araucania": "Araucanía", "biobio": "Biobío", "valparaiso": "Valparaíso",
    "los rios": "Los Ríos",
}


def norm_region(r):
    k = norm(r)
    return MAPA_REGION.get(k, str(r).strip() or "Desconocida")


def a_bool(v):
    k = norm(v)
    if k in {"1", "si", "true", "s"}:
        return 1
    if k in {"0", "no", "false", "n"}:
        return 0
    return 0  # vacio o raro -> no acepta (conservador)


def perfil(df, nombre):
    return {
        "archivo": nombre,
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "nulos_por_columna": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicados_fila": int(df.duplicated().sum()),
    }


def escribir_xgboost(matrices, etiquetas, columnas, clases):
    """Formato que exige el XGBoost integrado de SageMaker.

    train y validation: la etiqueta como numero en la PRIMERA columna y sin
    cabecera. test lleva cabecera y la etiqueta original, porque la usa
    sagemaker/4_probar.py para medir y necesita el nombre de la clase.

    Las columnas se llaman c000..cNNN y no con el termino del vocabulario, para
    que ninguna coma ni tilde del vocabulario rompa el CSV. El vocabulario real,
    en este mismo orden, queda en metadatos.json.
    """
    XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    indice = {c: i for i, c in enumerate(clases)}
    for nombre in ("train", "validation"):
        df = pd.DataFrame(matrices[nombre], columns=columnas)
        df.insert(0, "y", [indice[v] for v in etiquetas[nombre]])
        df.to_csv(XGBOOST_DIR / f"{nombre}.csv", index=False, header=False)
    df = pd.DataFrame(matrices["test"], columns=columnas)
    df.insert(0, "intencion", list(etiquetas["test"]))
    df.to_csv(XGBOOST_DIR / "test.csv", index=False)


def main():
    print(f"== preparar.py == umbral macro-F1 = {UMBRAL_F1}")
    clientes = pd.read_csv(RAW / "clientes.csv", dtype=str, keep_default_na=False)
    faq = pd.read_csv(RAW / "faq.csv", dtype=str, keep_default_na=False)
    tickets = pd.read_csv(RAW / "tickets.csv", dtype=str, keep_default_na=False)
    conversaciones = pd.read_csv(RAW / "conversaciones.csv", dtype=str, keep_default_na=False)

    antes = [perfil(clientes, "clientes"), perfil(faq, "faq"),
             perfil(tickets, "tickets"), perfil(conversaciones, "conversaciones")]
    resumen = {}

    # --- CLIENTES ---
    n0 = len(clientes)
    clientes = clientes.drop_duplicates(subset="cliente_id", keep="first")
    dup_cli = n0 - len(clientes)
    clientes["region"] = clientes["region"].map(norm_region)
    clientes["segmento"] = clientes["segmento"].map(
        lambda s: norm(s) if norm(s) in {"residencial", "pyme", "empresa"} else "desconocido")
    clientes["acepta_contacto"] = clientes["acepta_contacto"].map(a_bool)
    clientes["antiguedad_meses"] = pd.to_numeric(clientes["antiguedad_meses"], errors="coerce").fillna(0).astype(int)
    deuda = pd.to_numeric(clientes["deuda_vigente"], errors="coerce").fillna(0)
    deuda_neg = int((deuda < 0).sum())
    clientes["deuda_vigente"] = deuda.clip(lower=0).astype(int)  # deuda negativa no existe -> 0
    ids_validos = set(clientes["cliente_id"])
    clientes_out = clientes[["cliente_id", "fecha_alta", "plan", "tipo_servicio", "region",
                             "segmento", "antiguedad_meses", "deuda_vigente",
                             "canal_preferido", "acepta_contacto"]]
    resumen["clientes"] = {"duplicados_eliminados": dup_cli, "deuda_negativa_corregida": deuda_neg,
                           "filas_finales": len(clientes_out)}

    # --- FAQ ---
    n0 = len(faq)
    faq = faq[faq["vigente"] == "1"].copy()          # fuera no vigentes y nulos
    con_html = int(faq["respuesta"].str.contains("<").sum())
    faq["respuesta"] = faq["respuesta"].map(limpiar_html)       # fuera las etiquetas HTML
    faq["categoria"] = faq["categoria"].map(lambda s: norm(s).replace(" ", "_"))
    faq["intencion"] = faq["pregunta"].map(intencion_de_pregunta)   # no se usa la categoria: viene inconsistente
    corregidas = int((faq["categoria"].map(mapear_intencion) != faq["intencion"]).sum())
    faq["vistas_12m"] = pd.to_numeric(faq["vistas_12m"], errors="coerce").fillna(0).astype(int)
    faq_out = faq[["articulo_id", "pregunta", "respuesta", "categoria", "intencion", "vistas_12m", "ultima_edicion"]]
    resumen["faq"] = {"no_vigentes_descartados": n0 - len(faq_out),
                      "html_removido_de_la_respuesta": con_html,
                      "intencion_distinta_de_la_categoria": corregidas,
                      "filas_finales": len(faq_out)}

    # --- TICKETS ---
    n0 = len(tickets)
    f_ap = pd.to_datetime(tickets["fecha_apertura"], format="mixed", utc=True, errors="coerce")
    f_ci = pd.to_datetime(tickets["fecha_cierre"], format="mixed", utc=True, errors="coerce")
    cierre_invalido = int(((f_ci.notna()) & (f_ci < f_ap)).sum())
    f_ci = f_ci.where(~((f_ci.notna()) & (f_ci < f_ap)))     # cierre < apertura -> nulo
    espera = pd.to_numeric(tickets["tiempo_espera_seg"], errors="coerce")
    espera_neg = int((espera < 0).sum())
    espera_4h = int((espera > 4 * 3600).sum())
    espera = espera.where((espera >= 0) & (espera <= 4 * 3600))   # fuera de rango -> nulo
    cliente_tk = tickets["cliente_id"].where(tickets["cliente_id"].isin(ids_validos))
    huerfanos = int(tickets["cliente_id"].notna().sum() - cliente_tk.notna().sum())
    # limpiar() es la misma funcion que usa el servicio al predecir: asi el
    # modelo entrena y sirve sobre exactamente el mismo texto (RNF5)
    texto_mask = tickets["texto_consulta"].map(limpiar)
    pers_tk = int(tickets["texto_consulta"].map(tiene_pii).sum())
    intencion_tk = tickets["categoria"].map(mapear_intencion)

    tickets_out = pd.DataFrame({
        "ticket_id": tickets["ticket_id"],
        "cliente_id": cliente_tk,
        "canal": tickets["canal"].map(lambda s: norm(s)),
        "fecha_apertura": f_ap.dt.strftime("%Y-%m-%d %H:%M:%S%z"),
        "fecha_cierre": f_ci.dt.strftime("%Y-%m-%d %H:%M:%S%z"),
        "categoria": intencion_tk,
        "texto_consulta": texto_mask,            # enmascarado: nunca el crudo (RNF5)
        "tiempo_espera_seg": espera,
        "resuelto": pd.to_numeric(tickets["resuelto"], errors="coerce").fillna(0).astype(int),
        "csat": pd.to_numeric(tickets["csat"], errors="coerce"),
        "reabierto": pd.to_numeric(tickets["reabierto"], errors="coerce").fillna(0).astype(int),
    })
    resumen["tickets"] = {"filas": n0, "con_dato_personal_enmascarado": pers_tk,
                          "cliente_inexistente_a_null": huerfanos,
                          "espera_negativa_a_null": espera_neg, "espera_mayor_4h_a_null": espera_4h,
                          "cierre_antes_de_apertura_a_null": cierre_invalido}

    # --- CONVERSACIONES: se limpian y se perfilan, pero NO entrenan el modelo ---
    conv_dup = int(conversaciones.duplicated(subset=["conversacion_id"]).sum())
    resumen["conversaciones"] = {
        "duplicadas_eliminadas": conv_dup,
        "con_dato_personal": int(conversaciones["transcripcion"].map(tiene_pii).sum()),
        "usadas_entrenamiento": 0,
        "motivo_excluidas": "etiqueta puesta por el bot de reglas anterior; de 270 textos "
                            "compartidos con los tickets, 166 llevan etiqueta distinta "
                            "segun la fuente",
    }

    # --- DATASET DEL MODELO: solo tickets ---
    dataset = pd.DataFrame({"texto": texto_mask, "intencion": intencion_tk})
    dataset = dataset[dataset["texto"].str.len() >= 3].drop_duplicates().reset_index(drop=True)
    dataset["plantilla"] = dataset["texto"].map(plantilla)
    print(f"Dataset modelo: {len(dataset)} textos unicos en "
          f"{dataset['plantilla'].nunique()} plantillas distintas")
    print(dataset["intencion"].value_counts().to_string())

    # --- PARTICION POR PLANTILLA, EQUILIBRADA POR CLASE ---
    # StratifiedGroupKFold hace las dos cosas a la vez: ninguna plantilla cae en
    # los dos lados, y cada intencion queda repartida en la misma proporcion.
    # Con GroupShuffleSplit la prueba quedaba descuadrada (falla_conexion al 22%
    # cuando en el dataset es el 42%) y la linea base no era comparable.
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    i_tr, i_te = next(sgk.split(dataset, dataset["intencion"], groups=dataset["plantilla"]))
    entrenamiento, prueba = dataset.iloc[i_tr].copy(), dataset.iloc[i_te].copy()
    compartidas = set(entrenamiento["plantilla"]) & set(prueba["plantilla"])
    assert not compartidas, f"{len(compartidas)} plantillas quedaron en los dos lados"

    # validation sale de entrenamiento: la pide el Training Job de SageMaker
    sgk_val = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEMILLA)
    j_tr, j_val = next(sgk_val.split(entrenamiento, entrenamiento["intencion"],
                                     groups=entrenamiento["plantilla"]))
    train, validation = entrenamiento.iloc[j_tr], entrenamiento.iloc[j_val]
    print(f"entrenamiento {len(entrenamiento)} (train {len(train)} + validation {len(validation)}) "
          f"· prueba {len(prueba)}")

    # --- VECTORIZADOR: se ajusta SOLO con entrenamiento ---
    # Si se ajustara con todo, el vocabulario llevaria informacion de la prueba
    # y la metrica saldria mejor de lo que es.
    vectorizador = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                   sublinear_tf=True, strip_accents="unicode")
    M_entrenamiento = vectorizador.fit_transform(entrenamiento["texto"]).toarray()
    vocabulario = list(vectorizador.get_feature_names_out())
    columnas = [f"c{i:03d}" for i in range(len(vocabulario))]
    print(f"vocabulario: {len(vocabulario)} terminos (1-2 gramas, min_df=3)")

    M_train = vectorizador.transform(train["texto"]).toarray()
    M_validation = vectorizador.transform(validation["texto"]).toarray()
    M_prueba = vectorizador.transform(prueba["texto"]).toarray()

    # --- CLASIFICADOR ---
    modelo = LogisticRegression(C=2, max_iter=3000, class_weight="balanced")
    modelo.fit(M_entrenamiento, entrenamiento["intencion"].to_numpy())
    clases = [str(c) for c in modelo.classes_]

    pred = modelo.predict(M_prueba)
    y_te = prueba["intencion"].to_numpy()
    acc = accuracy_score(y_te, pred)
    macro_f1 = f1_score(y_te, pred, average="macro")
    reporte = classification_report(y_te, pred, output_dict=True, zero_division=0)
    # Linea base: responder siempre la clase mas frecuente
    mas_frecuente = entrenamiento["intencion"].mode()[0]
    base = float((y_te == mas_frecuente).mean())
    print(f"\naccuracy = {acc:.4f} | macro-F1 = {macro_f1:.4f} (umbral {UMBRAL_F1})")
    print(f"linea base (responder siempre '{mas_frecuente}') = {base:.4f}")

    # --- Guardar salidas ---
    clientes_out.to_csv(TRATADOS / "clientes.csv", index=False)
    faq_out.to_csv(TRATADOS / "faq.csv", index=False)
    tickets_out.to_csv(TRATADOS / "tickets.csv", index=False)

    despues = [perfil(clientes_out, "clientes"), perfil(faq_out, "faq"), perfil(tickets_out, "tickets")]
    (BASE / "perfil_antes.json").write_text(json.dumps(antes, indent=2, ensure_ascii=False))
    (BASE / "perfil_despues.json").write_text(json.dumps(despues, indent=2, ensure_ascii=False))
    (BASE / "limpieza_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

    escribir_xgboost(
        {"train": M_train, "validation": M_validation, "test": M_prueba},
        {"train": train["intencion"], "validation": validation["intencion"],
         "test": prueba["intencion"]},
        columnas, clases)

    metadatos = {
        "caso": CASO,
        "tipo": "multiclase",
        "version": f"{CASO}-{MODELO_VERSION}",
        "modelo_version": MODELO_VERSION,
        "entrenado_en": date.today().isoformat(),
        "algoritmo": "TF-IDF de palabras (1-2 gramas, min_df=3) + LogisticRegression "
                     "(class_weight=balanced). En EP3 el mismo CSV entrena un XGBoost "
                     "multi:softprob en SageMaker",
        "columnas": columnas,
        "vocabulario": vocabulario,
        "clases": clases,
        "particion": {"criterio": "por plantilla (digitos normalizados)",
                      "plantillas_entrenamiento": int(entrenamiento["plantilla"].nunique()),
                      "plantillas_prueba": int(prueba["plantilla"].nunique()),
                      "plantillas_compartidas": 0},
        "n_entrenamiento": int(len(entrenamiento)),
        "n_train_xgboost": int(len(train)),
        "n_validation_xgboost": int(len(validation)),
        "n_prueba": int(len(prueba)),
        "umbral_macro_f1": UMBRAL_F1,
        "metricas": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "linea_base_clase_mas_frecuente": round(base, 4),
            "por_clase": {k: {kk: round(vv, 3) for kk, vv in v.items()}
                          for k, v in reporte.items() if k in INTENCIONES},
            "matriz_confusion": {"clases": clases,
                                 "filas_real_columnas_predicha":
                                     confusion_matrix(y_te, pred, labels=clases).tolist()},
        },
    }
    joblib.dump(vectorizador, MODELO_DIR / "vectorizador.pkl")
    joblib.dump(modelo, MODELO_DIR / "modelo.pkl")
    (MODELO_DIR / "metadatos.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False))

    filas = [f"| accuracy | {acc:.4f} |", f"| macro-F1 | {macro_f1:.4f} |",
             f"| linea base (clase mas frecuente) | {base:.4f} |"]
    filas += [f"| F1 de {c} | {reporte[c]['f1-score']:.3f} |" for c in clases]
    (MODELO_DIR / "METRICAS.md").write_text(
        f"# Metricas del modelo · {CASO} · {MODELO_VERSION}\n\n"
        f"Conjunto de prueba: {len(prueba)} textos en {prueba['plantilla'].nunique()} plantillas, "
        f"separado por plantilla antes de entrenar (0 plantillas compartidas).\n"
        f"Variables: TF-IDF de {len(vocabulario)} terminos ajustado solo con entrenamiento.\n\n"
        f"| Metrica | Valor |\n|---|---|\n" + "\n".join(filas) + "\n")

    print("Guardado: tratados/*.csv, modelo/vectorizador.pkl, modelo/modelo.pkl, "
          "modelo/metadatos.json, modelo/METRICAS.md, modelo/xgboost/*.csv")

    if macro_f1 < UMBRAL_F1:
        print(f"ERROR: macro-F1 {macro_f1:.4f} < umbral {UMBRAL_F1}. No se construye la imagen.")
        raise SystemExit(1)
    print("OK: metrica sobre el umbral.")


if __name__ == "__main__":
    main()
