import csv
import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agent_logs.csv")


def inicializar_logs():
    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "pregunta",
                "respuesta",
                "herramientas_usadas",
                "latencia_segundos",
                "estado",
                "error",
                "longitud_respuesta"
            ])


def registrar_ejecucion(
    pregunta,
    respuesta,
    latencia,
    herramientas_usadas="sin_tool_detectada",
    estado="OK",
    error=""
):
    inicializar_logs()

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pregunta,
            respuesta,
            herramientas_usadas,
            round(latencia, 3),
            estado,
            error,
            len(respuesta) if respuesta else 0
        ])