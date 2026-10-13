# ia — Microservicio de IA (clasificador de intención)

Entrena y sirve un clasificador de intención del Caso 2 (telco).

## Endpoints
- `GET /health` → `{status, modelo_version}`
- `POST /predict` `{"texto": "..."}` → `{intencion, confianza, modelo_version, entrada_enmascarada}`

Antes de clasificar enmascara RUT, teléfono, correo y tarjeta (RNF5): el modelo
nunca ve datos personales.

## Tratamiento y entrenamiento
```bash
pip install -r requirements-dev.txt
python datos/preparar.py        # perfila, limpia, entrena; gate de métrica
```
Salidas: `modelo/modelo.pkl`, `modelo/modelo_meta.json` (van en la imagen) y
`datos/tratados/*.csv` (los usa el servicio de carga del repo dominio).
Umbral configurable con `UMBRAL_F1` (por defecto 0.65).

Modelo: TF-IDF de palabras (1-2 gramas) + LogisticRegression. Liviano, sin
PyTorch. Métrica: accuracy ≈ 0.75, macro-F1 ≈ 0.70.

## Probar y construir
```bash
pytest -q
docker build -t ia .
```
