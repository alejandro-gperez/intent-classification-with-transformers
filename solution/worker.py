#!/usr/bin/env python3
"""
worker.py — consume incoming_social_messages → clasifica → publica en cola de categoría

Flujo:
  RabbitMQ (incoming_social_messages)
      → predict(text)
      → publica en cola de categoría (tarjetas, pagos, transferencias,
                                      fondeo, divisas, cumplimiento, fraude)
"""
import json
import logging
import os
import sys
import time

import pika

from src.inference.predictor import predict

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Configuración (env vars con defaults seguros) ─────────────────────────────
RABBITMQ_HOST  = os.getenv("RABBITMQ_HOST",  "rabbitmq")
RABBITMQ_USER  = os.getenv("RABBITMQ_USER",  "bam")
RABBITMQ_PASS  = os.getenv("RABBITMQ_PASS",  "r3t0_analitico")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")

INPUT_QUEUE = "incoming_social_messages"
OUTPUT_QUEUES = [
    "tarjetas",
    "pagos",
    "transferencias",
    "fondeo",
    "divisas",
    "cumplimiento",
    "fraude",
]


# ── Conexión con reintentos ───────────────────────────────────────────────────
def connect(retries: int = 12, delay: int = 5) -> pika.BlockingConnection:
    """Intenta conectar a RabbitMQ con reintentos."""
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        virtual_host=RABBITMQ_VHOST,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=300,
    )
    for attempt in range(1, retries + 1):
        try:
            conn = pika.BlockingConnection(params)
            log.info("Conectado a RabbitMQ (intento %d/%d)", attempt, retries)
            return conn
        except Exception as exc:
            log.warning("Intento %d/%d — %s", attempt, retries, exc)
            if attempt == retries:
                log.error("No se pudo conectar a RabbitMQ tras %d intentos.", retries)
                raise
            time.sleep(delay)


# ── Declarar colas ────────────────────────────────────────────────────────────
def declare_queues(ch):
    """Declara la cola de entrada y las 7 colas de categoría."""
    ch.queue_declare(queue=INPUT_QUEUE, durable=True)
    for q in OUTPUT_QUEUES:
        ch.queue_declare(queue=q, durable=True)
    log.info("Colas declaradas OK — entrada: %s | salida: %s",
             INPUT_QUEUE, OUTPUT_QUEUES)


# ── Callback de procesamiento ─────────────────────────────────────────────────
def on_message(channel, method, properties, body):
    """
    Procesamiento de cada mensaje:
      1. Decodificar texto
      2. Clasificar con predict()
      3. Publicar resultado en la cola de categoría
      4. ACK / NACK
    """
    try:
        # 1. Decodificar
        raw = body.decode("utf-8")
        try:
            payload = json.loads(raw)
            text = payload.get("text", raw)
        except (json.JSONDecodeError, AttributeError):
            text = raw

        log.info("Mensaje recibido: %.90s%s", text, "..." if len(text) > 90 else "")

        # 2. Clasificar
        result   = predict(text)
        label    = result["label"]
        category = result["category"]

        # 3. Construir cuerpo de salida
        out = json.dumps(
            {"message": text, "label": label, "category": category},
            ensure_ascii=False,
        )

        # 4. Publicar en la cola correspondiente
        channel.basic_publish(
            exchange="",
            routing_key=category,
            body=out.encode("utf-8"),
            properties=pika.BasicProperties(delivery_mode=2),  # persistente
        )

        log.info(
            "Publicado → [%-15s] label=%-40s confidence=%.3f decision=%s",
            category, label, result["confidence"], result["decision"],
        )

        # ACK: mensaje procesado correctamente
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as exc:
        log.error("Error procesando mensaje: %s", exc, exc_info=True)
        # NACK sin requeue: evita loop infinito si el mensaje es inválido
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("========== Worker BAM iniciando ==========")
    log.info("Host: %s | Vhost: %s | Cola: %s",
             RABBITMQ_HOST, RABBITMQ_VHOST, INPUT_QUEUE)

    connection = connect()
    channel    = connection.channel()

    declare_queues(channel)

    # prefetch_count=1: procesar un mensaje a la vez (respeta CPU/mem del modelo)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=on_message)

    log.info("Worker listo. Esperando mensajes en '%s'... (Ctrl+C para salir)",
             INPUT_QUEUE)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        log.info("Worker detenido manualmente.")
        channel.stop_consuming()
    finally:
        if connection.is_open:
            connection.close()
        log.info("Conexión cerrada.")


if __name__ == "__main__":
    main()