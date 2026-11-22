import os

import pytest

from app.motores import MotorLocal, cargar_metadatos, crear_motor

META = cargar_metadatos()


def test_el_motor_local_devuelve_una_probabilidad_por_clase():
    motor = MotorLocal(META)
    proba = motor.probabilidades([[0.0] * len(META["columnas"])])
    assert len(proba) == 1
    assert len(proba[0]) == len(META["clases"])
    assert abs(sum(proba[0]) - 1.0) < 1e-6


def test_las_clases_del_modelo_son_las_de_los_metadatos():
    # Si se separan, /predict devuelve el nombre de otra intencion
    assert MotorLocal(META).clases == META["clases"]


def test_por_omision_el_motor_es_el_local():
    os.environ.pop("MOTOR", None)
    assert crear_motor(META).nombre == "local"


def test_un_motor_que_no_existe_falla_al_arrancar(monkeypatch):
    # Mejor que arranque mal a que responda con un modelo que no es el que se cree
    monkeypatch.setenv("MOTOR", "inventado")
    with pytest.raises(SystemExit):
        crear_motor(META)
