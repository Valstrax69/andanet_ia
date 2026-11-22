import json
from pathlib import Path

from app.caracteristicas import cargar_vectorizador, columnas, limpiar, vectorizar

META = json.loads((Path(__file__).resolve().parents[1] / "modelo" / "metadatos.json").read_text())


def test_el_dato_personal_no_llega_al_modelo():
    # RNF5: el modelo nunca ve el valor real
    t = limpiar("mi rut es 12.345.678-5, mi correo juan@mail.cl y mi fono +56 9 1234 5678")
    assert "12.345.678-5" not in t
    assert "juan@mail.cl" not in t
    assert "1234 5678" not in t
    assert "[RUT]" in t and "[EMAIL]" in t and "[TEL]" in t


def test_arregla_las_entidades_del_archivo():
    # En estos datos "&amp;" reemplaza a la letra "y"
    assert limpiar("necesito a&amp;uda") == "necesito ayuda"


def test_una_columna_por_termino_del_vocabulario():
    v = vectorizar("se me cayo internet")
    assert len(v) == len(META["columnas"]) == len(META["vocabulario"])


def test_el_vocabulario_guardado_es_el_del_vectorizador():
    # Si estos dos se separan, el endpoint de SageMaker recibe las columnas
    # en otro orden y predice cualquier cosa
    assert columnas() == META["vocabulario"]


def test_la_misma_entrada_da_el_mismo_vector():
    texto = "quiero cambiar mi plan"
    assert vectorizar(texto) == vectorizar(texto)


def test_texto_sin_palabras_conocidas_no_rompe():
    v = vectorizar("xyzzy qwertz")
    assert len(v) == len(META["columnas"])
    assert sum(v) == 0.0
