# Reto Analítico Banco Agromercantil 
## Informe Ejecutivo - Grupo AAC
Integrantes: Alejandra Sierra, Alejandro Pérez, Camila Sandoval

---

## Resumen del proyecto

El banco recibe miles de mensajes diarios por distintos canales: app, web, email y redes sociales. Antes de este sistema, un equipo de analistas los leía uno por uno y los redirigía manualmente al área correcta. Ese proceso no escala, genera demoras y no aprovecha el tiempo de las personas que lo hacen.

Este proyecto construye un clasificador automático que lee cada mensaje, identifica la intención del cliente entre 77 posibles, y lo envía a la cola correcta en cuestión de segundos. Todo corre dentro de contenedores Docker conectados a RabbitMQ, sin depender de servicios externos.

---

## De limpieza a producción: cómo llegamos hasta acá

El trabajo se dividió en tres partes que se construyeron una sobre la otra:

**Parte 1 — Datos y modelo.** Empezamos explorando el dataset: revisamos nulos, duplicados, distribución de clases y calidad general del texto. Con eso claro, preparamos los datos para entrenar: codificamos las etiquetas, hicimos el split de entrenamiento y validación de forma estratificada para respetar la distribución de las 77 clases, tokenizamos los textos con el tokenizador de DistilBERT y calculamos pesos por clase para compensar el desbalance. Luego hicimos el fine-tuning y evaluamos con F1.

**Parte 2 — Lógica de predicción.** Con el modelo entrenado, el siguiente paso fue convertirlo en algo robusto para producción. Se construyó una función `predict()` que no solo devuelve la etiqueta más probable, sino que evalúa qué tan seguro está el modelo antes de tomar una decisión. Si la confianza es baja o las dos opciones principales están muy cerca, el sistema lo marca para revisión en vez de adivinar.

**Parte 3 — Worker e integración.** La última pieza conecta todo: un worker dockerizado que consume mensajes de RabbitMQ, llama a `predict()`, y publica el resultado en la cola de la categoría correspondiente. El stack completo se levanta con un solo comando.

---

## Por qué elegimos DistilBERT

Evaluamos tres enfoques: un modelo de machine learning tradicional, fine-tuning de un transformer, y un flujo con LLM. La decisión fue DistilBERT por varias razones concretas:

| Criterio | ML tradicional | DistilBERT (elegido) | LLM workflow |
|---|---|---|---|
| F1 esperado (77 clases) | 0.70–0.80 | **0.85–0.90** | 0.88–0.92 |
| Latencia por mensaje | <10ms | ~80ms CPU / ~15ms GPU | 500ms–2s |
| Costo operativo | Bajo | Bajo (modelo local) | Alto (API externa) |
| Independencia de red | ✅ | ✅ | ❌ |
| Corre en CPU (Docker) | ✅ | ✅ | Depende |

DistilBERT es una versión compacta de BERT que mantiene buena precisión pero corre bien en CPU, lo cual era importante porque el worker corre dentro de Docker sin garantía de GPU. Además, Hugging Face tiene todo el ecosistema listo — tokenizador, modelo preentrenado, Trainer — así que en el tiempo disponible (5 días) era la opción más viable de implementar correctamente.

El LLM hubiera dado F1 marginalmente más alto, pero depende de una API externa de pago, tiene latencia variable y agrega una dependencia de red que complica el despliegue local. El ML tradicional no alcanza el F1 necesario con 77 clases desbalanceadas.

Una ventaja adicional: el desbalance de clases se manejó con `class_weights` directamente en la función de pérdida, algo que DistilBERT permite hacer de forma limpia y que en modelos tradicionales suele requerir técnicas más complejas como oversampling o SMOTE.

**Resultado obtenido:** F1 weighted **0.8539** · F1 macro **0.8557** · Accuracy **0.86** sobre 2,064 ejemplos de validación.

---

## Pipeline de entrenamiento

El proceso de preparar los datos fue tan importante como el entrenamiento en sí. DistilBERT no entiende texto directamente — necesita que cada mensaje esté convertido en secuencias numéricas (tokens) con un formato específico. Antes de llegar ahí, el dataset necesitaba estar limpio.

Se eliminaron nulos y duplicados para no introducir ruido en el entrenamiento. Luego se codificaron las 77 etiquetas como números, se hizo el split estratificado (80% entrenamiento, 20% validación) para que todas las clases estuvieran representadas en ambas particiones, y se tokenizó con `MAX_LENGTH=256`, suficiente para cubrir la mayoría de los mensajes de clientes sin truncar información relevante.

El desbalance de clases era visible: algunas intenciones tenían muchos más ejemplos que otras. Sin corrección, el modelo tiende a ignorar las clases pequeñas. Se calcularon pesos inversamente proporcionales a la frecuencia de cada clase y se aplicaron en la función de pérdida durante el entrenamiento.

**Progresión por época:**

| Época | Training Loss | Validation Loss | F1 Weighted | F1 Macro |
|---|---|---|---|---|
| 1 | 3.2958 | 2.9744 | 0.4650 | 0.4720 |
| 2 | 2.0829 | 1.8355 | 0.7561 | 0.7591 |
| 3 | 1.4419 | 1.2500 | 0.8224 | 0.8238 |
| 4 | 1.0858 | 0.9856 | 0.8517 | 0.8527 |
| 5 | 0.9481 | 0.9094 | **0.8539** | **0.8557** |

El modelo mejoró consistentemente sin señales de sobreajuste — la validation loss bajó en todas las épocas.

---

## Lógica de predicción y safety net

Una observación importante al poner el modelo en producción: aunque el F1 global es bueno, en muchos mensajes el modelo está entre 40–50% seguro. Un F1 alto no significa que siempre esté muy confiado — significa que acierta con frecuencia, pero no necesariamente con margen amplio.

Para manejar eso, `predict()` implementa una capa de decisión antes de publicar el resultado:

```
si confianza >= 0.70:
    → high_confidence: publicar top-1 directamente

si confianza >= 0.40:
    si diferencia entre top-1 y top-2 < 0.15:
        → ambiguous_used_top2: las dos opciones están muy cerca, usar top-2
    si no:
        → medium_confidence: publicar top-1

si confianza < 0.40:
    → low_confidence: publicar top-1 pero marcar para revisión humana
```

Cada mensaje publicado incluye el campo `decision`, que le dice al equipo de atención con qué nivel de certeza llegó ese mensaje. Eso permite priorizar revisión manual en los casos donde el modelo dudó.

---

## ¿Dónde vive el modelo?

El modelo vive dentro de la imagen Docker. Cuando se construye la imagen, los archivos del modelo se copian adentro y el worker los carga al arrancar. No necesita descargar nada ni conectarse a ningún servicio externo para funcionar.

```
Imagen Docker
└── /app/
    ├── worker.py
    ├── src/inference/
    │   ├── predictor.py       ← función predict() con safety net
    │   ├── config.py          ← thresholds de confianza
    │   └── preprocessing.py   ← limpieza de texto
    └── model/
        ├── model.safetensors  ← pesos del modelo (~250MB)
        ├── tokenizer.json
        ├── config.json
        └── label_encoder.pkl  ← mapeo label_id → nombre de intención
```

Esto garantiza que la misma imagen siempre use el mismo modelo. La limitación es que si se reentrena el modelo, hay que reconstruir la imagen. En un entorno de producción más maduro se usaría un registry como MLflow o S3, pero para el alcance de este proyecto la solución embebida es la correcta.

---

## ¿Cómo se actualiza el modelo?

Si se obtienen datos nuevos o se quiere mejorar el modelo, el proceso es:

1. Correr `cleaning/model-training.ipynb` con los datos actualizados. Esto genera archivos nuevos en `model/`.
2. Reconstruir y reiniciar el worker:

```bash
docker compose -f deploy/compose.yml -f solution/compose.override.yml build worker
docker compose -f deploy/compose.yml -f solution/compose.override.yml up -d worker
```

Mientras el worker se reinicia, los mensajes que lleguen no se pierden — quedan esperando en la cola `incoming_social_messages` y se procesan en cuanto el nuevo worker está listo. Esto es una ventaja directa de la arquitectura event-driven: el clasificador y el canal de entrada están desacoplados.

---

## ¿Qué pasa si el volumen de mensajes se duplica?

El worker actual procesa un mensaje a la vez. Si el volumen crece mucho, los mensajes se acumulan en la cola — RabbitMQ los sostiene sin problema — pero el tiempo de respuesta aumenta.

La solución es levantar más instancias del worker:

```bash
docker compose -f deploy/compose.yml -f solution/compose.override.yml up -d --scale worker=3
```

Con eso, tres workers procesan mensajes en paralelo desde la misma cola y RabbitMQ los distribuye automáticamente. No requiere cambios en el código. El cuello de botella es la inferencia del modelo (no el broker), y agregar workers es la forma directa de resolverlo.

Cada instancia de DistilBERT ocupa aproximadamente 250MB en memoria. Con tres instancias son ~750MB, lo cual es manejable en cualquier servidor moderno.

---

## ¿Cómo se monitorea en producción?

**RabbitMQ Management UI** (`http://localhost:15672`): permite ver en tiempo real cuántos mensajes hay en cada cola, a qué velocidad se procesan, y si el worker está activo. Si la cola de entrada crece sin parar, algo está mal con el worker.

**Logs del worker:** cada mensaje deja un registro con su etiqueta, categoría, confianza y tipo de decisión:

```bash
# Ver mensajes de baja confianza
docker logs bam_worker -f | grep "low_confidence"
```

**Distribución de categorías:** si de repente el 80% de los mensajes van a `fraude` cuando históricamente era el 15%, hay un problema — ya sea en el modelo o en el mapeo de etiquetas.

| Señal | Causa probable |
|---|---|
| Cola de entrada crece sin parar | Worker caído o muy lento |
| Muchos mensajes con `low_confidence` | Mensajes fuera de lo que el modelo conoce |
| Una categoría recibe casi todo el tráfico | Bug en el mapeo label → categoría |
| F1 baja en revisión humana | El modelo necesita reentrenarse |

La métrica que realmente importa al negocio no es el F1 del notebook, sino cuántos mensajes llegan al equipo correcto. Eso solo se puede medir comparando predicciones contra revisiones humanas en producción.

---

## Levantar el stack completo

```bash
docker compose -f deploy/compose.yml -f solution/compose.override.yml up -d
```

Ese comando levanta RabbitMQ, el producer de mensajes sintéticos y el worker. Los mensajes empiezan a fluir automáticamente hacia las colas de categoría.
