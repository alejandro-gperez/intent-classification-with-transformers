# Reto Analítico 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Bienvenidos al repositorio oficial del **Reto Analítico 2026**, organizado por el **Departamento de Capacidades Analíticas** del Banco Agromercantil.

El reto consiste en construir el corazón de un sistema automático de clasificación y ruteo de mensajes de servicio al cliente: un clasificador que asigna cada mensaje a una de **77 intenciones** y, en producción, lo enruta a la cola del equipo de atención correspondiente.

> [!IMPORTANT]
> Revisa las instrucciones detalladas en `docs/instructions.html`.

---

## El dataset

| Partición | Registros | Etiquetas |
|---|---|---|
| `data/train.csv` | 10,329 | ✓ incluidas |
| `data/test.csv` | 2,771 | — hold-out del organizador |

Los mensajes están en inglés. No hay metadatos: ni canal de origen, ni fecha, ni información del cliente. El clasificador decide a partir del texto únicamente.

Las 77 etiquetas se agrupan en 7 categorías definidas en `data/categories.json`:

`tarjetas` · `pagos` · `transferencias` · `fondeo` · `divisas` · `cumplimiento` · `fraude`

---

## La tarea

El clasificador debe predecir la columna `label` para cada mensaje de `data/test.csv`. La evaluación se realiza con **F1 score ponderado** sobre las 77 clases.

El equipo puede implementar el clasificador con cualquiera de estos enfoques—o una combinación propia:

- Machine learning tradicional
- Transfer learning con un Transformer
- LLM workflow

---

## Despliegue

El entorno de simulación usa **RabbitMQ** como broker de mensajes.

> [!NOTE]
> Docker o Docker Desktop debe estar instalado antes de levantar el stack.

```bash
# Levantar broker + productor de mensajes sintéticos
docker compose -f deploy/compose.yml up -d

# Levantar el stack completo (broker + productor + worker del equipo)
docker compose -f deploy/compose.yml -f solution/compose.override.yml up -d
```

El portal de administración de RabbitMQ está disponible en [http://localhost:15672](http://localhost:15672) (`bam` / `r3t0_analitico`).

El worker consume de `incoming_social_messages` y publica en las siete colas de categoría con el siguiente contrato:

```json
{
  "message": "<texto original del cliente>",
  "label": "<una de las 77 etiquetas predichas>",
  "category": "<una de las 7 categorías>"
}
```

---

## Estructura del entregable

```
solution/
├── worker.py             # consume → clasifica → publica
├── Dockerfile
├── compose.override.yml  # define el servicio "worker", con depends_on: rabbitmq
├── requirements.txt
└── README.md             # informe ejecutivo
```

> [!IMPORTANT]
> El archivo `data/test.csv` debe contener la columna `label` completa al momento de la entrega. El organizador usa ese archivo para calcular el F1 score contra las etiquetas reales.

> [!IMPORTANT]
> El `solution/README.md` es un entregable evaluado. Debe justificar las decisiones sobre el mecanismo de clasificación: dónde vive el modelo, cómo se actualiza, qué ocurre si el volumen de mensajes se duplica, y cómo se monitorea en producción.

---

## Criterios de evaluación

| Entregable | Qué se mide |
|---|---|
| `data/test.csv` con columna `label` | F1 score ponderado sobre las 77 etiquetas |
| Stack dockerizado (`solution/`) | El worker conecta, consume y publica con el contrato definido |
| `solution/README.md` | Claridad y profundidad del análisis de la solución |

---

Departamento de Capacidades Analíticas  
Banco Agromercantil

