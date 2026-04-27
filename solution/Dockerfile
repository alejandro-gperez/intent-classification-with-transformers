# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — Worker BAM
# Build context: raíz del repositorio  (context: . en compose.override.yml)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Dependencias del sistema necesarias para compilar extensiones de torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Instalar dependencias Python (capa cacheada) ──────────────────────────────
# Copiamos solo requirements.txt primero para aprovechar el cache de Docker:
# si el código cambia pero requirements.txt no, esta capa no se reconstruye.
COPY solution/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Copiar estructura del repo manteniendo rutas relativas ───────────────────
#
# predictor.py usa:
#   BASE_DIR = os.path.dirname(__file__)          →  /repo/solution/src/inference
#   MODEL_DIR = os.path.abspath(BASE_DIR + "../../..model")  →  /repo/model  ✓
#   DATA_DIR  = os.path.abspath(BASE_DIR + "../../../data")  →  /repo/data   ✓
#
# Por eso respetamos la misma jerarquía de carpetas del repositorio.

# Código de inferencia (predictor, config, preprocessing)
COPY solution/src/ /repo/solution/src/

# Modelo DistilBERT fine-tuned (gitignored, debe existir localmente)
COPY model/ /repo/model/

# Mapeo label → categoría (fuente de verdad)
COPY data/categories.json /repo/data/categories.json

# Worker (punto de entrada)
COPY solution/worker.py /repo/solution/worker.py

# ── Directorio de trabajo: solution/ para que los imports funcionen ───────────
# `from src.inference.predictor import predict` busca src/ relativo al CWD
WORKDIR /repo/solution

CMD ["python", "worker.py"]