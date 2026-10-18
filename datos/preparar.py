"""
preparar.py - Tratamiento de datos y entrenamiento del modelo de intencion.

Lee los CSV crudos del Caso 2 (telco), los perfila, los limpia con reglas
documentadas, enmascara datos personales, arma el dataset de entrenamiento,
entrena un clasificador de intencion y guarda:
  - datos/tratados/{clientes,faq,tickets}.csv  -> los usa el servicio de carga
  - modelo/modelo.pkl + modelo/modelo_meta.json -> van dentro de la imagen de ia

Si la metrica (macro-F1) queda bajo el umbral, el script termina con codigo 1
para que el pipeline no construya la imagen.

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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- Rutas ---
BASE = Path(__file__).resolve().parent          # .../ia/datos
sys.path.insert(0, str(BASE.parent))            # para importar app.anonimizar
from app.anonimizar import enmascarar, tiene_pii  # noqa: E402
RAW = BASE / "raw"
TRATADOS = BASE / "tratados"
MODELO_DIR = BASE.parent / "modelo"             # .../ia/modelo
TRATADOS.mkdir(parents=True, exist_ok=True)
MODELO_DIR.mkdir(parents=True, exist_ok=True)

MODELO_VERSION = "1.0.0"
UMBRAL_F1 = float(os.getenv("UMBRAL_F1", "0.65"))

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


# Intenciones de conversaciones que no son etiquetas confiables (fallas del bot viejo)
SIN_ETIQUETA = {"desconocida", "unknown", ""}

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
    texto_mask = tickets["texto_consulta"].map(enmascarar)
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

    # --- CONVERSACIONES (solo para entrenamiento, no van a la base) ---
    conv = conversaciones.copy()
    conv_dup = int(conv.duplicated(subset=["conversacion_id"]).sum())
    conv = conv.drop_duplicates(subset=["conversacion_id"], keep="first")
    raw_int = conv["intencion_detectada"].map(norm)
    conv_sin = int(raw_int.isin(SIN_ETIQUETA).sum())
    conv = conv[~raw_int.isin(SIN_ETIQUETA)].copy()          # fuera las sin etiqueta confiable
    def primer_turno(t):
        t = str(t).replace("|", " ")
        m = re.search(r"cliente\s*:\s*(.*?)(?:asistente\s*:|agente\s*:|bot\s*:|$)", t, re.IGNORECASE | re.DOTALL)
        return enmascarar(m.group(1) if m else t)[:500]
    conv["texto"] = conv["transcripcion"].map(primer_turno)
    conv["intencion"] = conv["intencion_detectada"].map(mapear_intencion)
    resumen["conversaciones"] = {"duplicadas_eliminadas": conv_dup,
                                 "con_dato_personal": int(conversaciones["transcripcion"].map(tiene_pii).sum()),
                                 "sin_etiqueta_descartadas": conv_sin,
                                 "usadas_entrenamiento": len(conv)}

    # --- DATASET DEL MODELO: tickets (texto) + conversaciones (primer turno) ---
    ds_tk = pd.DataFrame({"texto": texto_mask, "intencion": intencion_tk})
    ds_cv = conv[["texto", "intencion"]]
    dataset = pd.concat([ds_tk, ds_cv], ignore_index=True)
    dataset = dataset[dataset["texto"].str.len() >= 3].drop_duplicates()
    print("Dataset modelo:", len(dataset), "filas")
    print(dataset["intencion"].value_counts().to_string())

    X_tr, X_te, y_tr, y_te = train_test_split(
        dataset["texto"], dataset["intencion"], test_size=0.2,
        random_state=42, stratify=dataset["intencion"])

    # TF-IDF de palabras (1-2 gramas). Acierta las consultas tipicas del caso.
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("clf", LogisticRegression(C=2, max_iter=3000, class_weight="balanced")),
    ])
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    acc = accuracy_score(y_te, pred)
    macro_f1 = f1_score(y_te, pred, average="macro")
    reporte = classification_report(y_te, pred, output_dict=True, zero_division=0)
    print(f"\naccuracy = {acc:.3f} | macro-F1 = {macro_f1:.3f} (umbral {UMBRAL_F1})")

    # --- Guardar salidas ---
    clientes_out.to_csv(TRATADOS / "clientes.csv", index=False)
    faq_out.to_csv(TRATADOS / "faq.csv", index=False)
    tickets_out.to_csv(TRATADOS / "tickets.csv", index=False)

    despues = [perfil(clientes_out, "clientes"), perfil(faq_out, "faq"), perfil(tickets_out, "tickets")]
    (BASE / "perfil_antes.json").write_text(json.dumps(antes, indent=2, ensure_ascii=False))
    (BASE / "perfil_despues.json").write_text(json.dumps(despues, indent=2, ensure_ascii=False))
    (BASE / "limpieza_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

    meta = {
        "modelo_version": MODELO_VERSION,
        "entrenado_en": date.today().isoformat(),
        "algoritmo": "TF-IDF de palabras (1-2 gramas) + LogisticRegression (class_weight=balanced)",
        "clases": sorted(dataset["intencion"].unique().tolist()),
        "n_entrenamiento": int(len(X_tr)),
        "n_prueba": int(len(X_te)),
        "umbral_macro_f1": UMBRAL_F1,
        "metricas": {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
                     "por_clase": {k: {kk: round(vv, 3) for kk, vv in v.items()}
                                   for k, v in reporte.items() if k in INTENCIONES}},
    }
    joblib.dump(modelo, MODELO_DIR / "modelo.pkl")
    (MODELO_DIR / "modelo_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("Guardado: tratados/*.csv, modelo/modelo.pkl, modelo/modelo_meta.json")

    if macro_f1 < UMBRAL_F1:
        print(f"ERROR: macro-F1 {macro_f1:.3f} < umbral {UMBRAL_F1}. No se construye la imagen.")
        raise SystemExit(1)
    print("OK: metrica sobre el umbral.")


if __name__ == "__main__":
    main()
