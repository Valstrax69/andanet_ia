"""Enmascarado de datos personales (RNF5). El modelo nunca ve datos crudos."""
import re

RE_TARJETA = re.compile(r"\b(?:\d[ -]?){13,16}\b")
RE_RUT = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b")
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_TEL = re.compile(r"(?:\+?56)?\s?9\s?\d{4}\s?\d{4}")
RE_FOOTER = re.compile(r"enviado desde.*$", re.IGNORECASE | re.DOTALL)


def tiene_pii(texto):
    t = str(texto)
    return bool(RE_TARJETA.search(t) or RE_RUT.search(t) or RE_EMAIL.search(t) or RE_TEL.search(t))


def enmascarar(texto):
    t = str(texto).replace("\\n", " ").replace("\n", " ")
    t = RE_FOOTER.sub(" ", t)
    t = RE_TARJETA.sub("[TARJETA]", t)
    t = RE_RUT.sub("[RUT]", t)
    t = RE_EMAIL.sub("[EMAIL]", t)
    t = RE_TEL.sub("[TEL]", t)
    return re.sub(r"\s+", " ", t).strip()
