import csv
import os
import time

from agent import agent, mostrar_ultima_respuesta, obtener_herramientas_usadas

LOG_DIR = "logs"
RESULTS_FILE = os.path.join(LOG_DIR, "evaluation_results.csv")
CONSISTENCY_FILE = os.path.join(LOG_DIR, "consistency_results.csv")

ESPERA_ENTRE_PRUEBAS = 20
MAX_REINTENTOS = 3


def crear_carpeta_logs():
    os.makedirs(LOG_DIR, exist_ok=True)


def calcular_precision(respuesta, palabras_clave):
    respuesta_lower = respuesta.lower()
    aciertos = 0

    for palabra in palabras_clave:
        if palabra.lower() in respuesta_lower:
            aciertos += 1

    if len(palabras_clave) == 0:
        return 0

    return round((aciertos / len(palabras_clave)) * 100, 2)


def ejecutar_consulta(pregunta, thread_id):
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            inicio = time.perf_counter()

            resultado = agent.invoke(
                {"messages": [{"role": "user", "content": pregunta}]},
                config={"configurable": {"thread_id": thread_id}}
            )

            fin = time.perf_counter()
            latencia = round(fin - inicio, 3)

            respuesta = mostrar_ultima_respuesta(resultado)
            herramientas = obtener_herramientas_usadas(resultado)

            return respuesta, herramientas, latencia

        except Exception as e:
            error = str(e)

            if "Too many requests" in error:
                espera = intento * 60
                print(f"Límite de GitHub alcanzado. Esperando {espera} segundos...")
                time.sleep(espera)
            else:
                raise e

    raise Exception("No se pudo ejecutar la consulta por límite de solicitudes de GitHub Models.")


def pruebas_precision():
    casos = [
        {
            "pregunta": "¿Cuánto dura la garantía de Samsung?",
            "palabras_clave": ["12 meses", "garantía", "Samsung"],
        },
        {
            "pregunta": "¿Cuánto demora el despacho a regiones?",
            "palabras_clave": ["3", "5", "días", "regiones"],
        },
        {
            "pregunta": "Si un producto vale 699990 y tiene 15% de descuento, cuánto queda?",
            "palabras_clave": ["594", "descuento", "15"],
        },
        {
            "pregunta": "Tengo un reclamo y necesito ayuda humana",
            "palabras_clave": ["resumen", "agente", "especializado"],
        },
    ]

    with open(RESULTS_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "pregunta",
            "respuesta",
            "herramientas_usadas",
            "latencia_segundos",
            "palabras_clave_esperadas",
            "precision_porcentaje",
            "estado"
        ])

        for i, caso in enumerate(casos, start=1):
            try:
                respuesta, herramientas, latencia = ejecutar_consulta(
                    caso["pregunta"],
                    thread_id=f"precision-test-{i}"
                )

                precision = calcular_precision(respuesta, caso["palabras_clave"])

                writer.writerow([
                    caso["pregunta"],
                    respuesta,
                    herramientas,
                    latencia,
                    ", ".join(caso["palabras_clave"]),
                    precision,
                    "OK"
                ])

                print(f"Prueba precisión {i}: {precision}%")
                time.sleep(ESPERA_ENTRE_PRUEBAS)

            except Exception as e:
                writer.writerow([
                    caso["pregunta"],
                    "",
                    "",
                    0,
                    ", ".join(caso["palabras_clave"]),
                    0,
                    f"ERROR: {str(e)}"
                ])

                print(f"Error en prueba precisión {i}: {e}")


def pruebas_consistencia():
    pares = [
        {
            "pregunta_1": "¿Cuánto dura la garantía de Samsung?",
            "pregunta_2": "¿Qué garantía tienen los productos Samsung?",
            "palabras_clave": ["12 meses", "garantía", "Samsung"],
        },
        {
            "pregunta_1": "¿Cuánto demora el despacho a regiones?",
            "pregunta_2": "¿En cuántos días llega un pedido a regiones?",
            "palabras_clave": ["3", "5", "días", "regiones"],
        },
    ]

    with open(CONSISTENCY_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "pregunta_1",
            "respuesta_1",
            "pregunta_2",
            "respuesta_2",
            "palabras_clave_esperadas",
            "consistencia_porcentaje",
            "estado"
        ])

        for i, par in enumerate(pares, start=1):
            try:
                respuesta_1, herramientas_1, latencia_1 = ejecutar_consulta(
                    par["pregunta_1"],
                    thread_id=f"consistency-test-{i}-a"
                )

                time.sleep(ESPERA_ENTRE_PRUEBAS)

                respuesta_2, herramientas_2, latencia_2 = ejecutar_consulta(
                    par["pregunta_2"],
                    thread_id=f"consistency-test-{i}-b"
                )

                precision_1 = calcular_precision(respuesta_1, par["palabras_clave"])
                precision_2 = calcular_precision(respuesta_2, par["palabras_clave"])

                consistencia = round((precision_1 + precision_2) / 2, 2)

                writer.writerow([
                    par["pregunta_1"],
                    respuesta_1,
                    par["pregunta_2"],
                    respuesta_2,
                    ", ".join(par["palabras_clave"]),
                    consistencia,
                    "OK"
                ])

                print(f"Prueba consistencia {i}: {consistencia}%")
                time.sleep(ESPERA_ENTRE_PRUEBAS)

            except Exception as e:
                writer.writerow([
                    par["pregunta_1"],
                    "",
                    par["pregunta_2"],
                    "",
                    ", ".join(par["palabras_clave"]),
                    0,
                    f"ERROR: {str(e)}"
                ])

                print(f"Error en prueba consistencia {i}: {e}")


def main():
    crear_carpeta_logs()

    print("Ejecutando pruebas de precisión...")
    pruebas_precision()

    print("\nEjecutando pruebas de consistencia...")
    pruebas_consistencia()

    print("\nPruebas finalizadas.")
    print(f"Resultados guardados en: {RESULTS_FILE}")
    print(f"Resultados guardados en: {CONSISTENCY_FILE}")


if __name__ == "__main__":
    main()