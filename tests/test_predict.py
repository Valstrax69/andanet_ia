from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
CLASES = {"facturacion", "falla_conexion", "cambio_plan", "visita", "anulacion", "otra"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["modelo_version"]


def test_predict_devuelve_prediccion_valida():
    r = client.post("/predict", json={"texto": "se me cayo internet y no tengo conexion"})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["intencion"] in CLASES
    assert 0.0 <= cuerpo["confianza"] <= 1.0
    assert cuerpo["modelo_version"]


def test_predict_enmascara_datos_personales():
    r = client.post("/predict", json={"texto": "mi numero es +56 9 1234 5678 llamenme"})
    assert r.status_code == 200
    assert "[TEL]" in r.json()["entrada_enmascarada"]
    assert "1234 5678" not in r.json()["entrada_enmascarada"]


def test_predict_rechaza_texto_vacio():
    r = client.post("/predict", json={"texto": ""})
    assert r.status_code == 422
