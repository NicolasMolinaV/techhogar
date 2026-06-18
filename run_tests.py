import csv
import os
import pandas as pd

LOG_DIR = "logs"
AGENT_LOGS = os.path.join(LOG_DIR, "agent_logs.csv")
RESULTS_FILE = os.path.join(LOG_DIR, "evaluation_results.csv")
SUMMARY_FILE = os.path.join(LOG_DIR, "metrics_summary.csv")


def calcular_precision(respuesta, palabras_clave):
    respuesta_lower = str(respuesta).lower()
    aciertos = 0

    for palabra in palabras_clave:
        if palabra.lower() in respuesta_lower:
            aciertos += 1

    if len(palabras_clave) == 0:
        return 0

    return round((aciertos / len(palabras_clave)) * 100, 2)


def evaluar_precision_desde_logs():
    df = pd.read_csv(AGENT_LOGS)

    casos = [
        {
            "tipo": "garantia",
            "buscar": "garantía",
            "palabras_clave": ["12 meses", "garantía"],
        },
        {
            "tipo": "descuento",
            "buscar": "descuento",
            "palabras_clave": ["594", "15", "descuento"],
        },
        {
            "tipo": "reclamo",
            "buscar": "reclamo",
            "palabras_clave": ["agente", "especializado"],
        },
        {
            "tipo": "caso_multiple",
            "buscar": "garantía, despacho",
            "palabras_clave": ["garantía", "despacho", "12 meses", "3", "5"],
        },
    ]

    with open(RESULTS_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "tipo",
            "pregunta",
            "respuesta",
            "palabras_clave_esperadas",
            "precision_porcentaje",
            "estado"
        ])

        for caso in casos:
            fila_encontrada = None

            for _, fila in df.iterrows():
                pregunta = str(fila["pregunta"]).lower()

                if caso["tipo"] == "garantia" and "garantía" in pregunta:
                    fila_encontrada = fila
                    break

                if caso["tipo"] == "descuento" and "descuento" in pregunta:
                    fila_encontrada = fila
                    break

                if caso["tipo"] == "reclamo" and "reclamo" in pregunta:
                    fila_encontrada = fila
                    break

                if caso["tipo"] == "caso_multiple" and "garantía" in pregunta and "despacho" in pregunta:
                    fila_encontrada = fila
                    break

            if fila_encontrada is not None:
                precision = calcular_precision(
                    fila_encontrada["respuesta"],
                    caso["palabras_clave"]
                )

                writer.writerow([
                    caso["tipo"],
                    fila_encontrada["pregunta"],
                    fila_encontrada["respuesta"],
                    ", ".join(caso["palabras_clave"]),
                    precision,
                    "OK"
                ])

                print(f"Precisión {caso['tipo']}: {precision}%")

            else:
                writer.writerow([
                    caso["tipo"],
                    "",
                    "",
                    ", ".join(caso["palabras_clave"]),
                    0,
                    "NO ENCONTRADO"
                ])

                print(f"Precisión {caso['tipo']}: no encontrado")


def generar_resumen_metricas():
    df = pd.read_csv(AGENT_LOGS)

    total_consultas = len(df)
    latencia_promedio = round(df["latencia_segundos"].mean(), 3)
    latencia_maxima = round(df["latencia_segundos"].max(), 3)
    errores = len(df[df["estado"] == "ERROR"])
    tasa_error = round((errores / total_consultas) * 100, 2) if total_consultas > 0 else 0
    longitud_promedio = round(df["longitud_respuesta"].mean(), 2)

    consulta_lenta = df.loc[df["latencia_segundos"].idxmax()]["pregunta"]

    with open(SUMMARY_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["metrica", "valor"])
        writer.writerow(["total_consultas", total_consultas])
        writer.writerow(["latencia_promedio", latencia_promedio])
        writer.writerow(["latencia_maxima", latencia_maxima])
        writer.writerow(["errores", errores])
        writer.writerow(["tasa_error", tasa_error])
        writer.writerow(["longitud_promedio_respuesta", longitud_promedio])
        writer.writerow(["consulta_mas_lenta", consulta_lenta])

    print("\nResumen de métricas:")
    print(f"Total consultas: {total_consultas}")
    print(f"Latencia promedio: {latencia_promedio} segundos")
    print(f"Latencia máxima: {latencia_maxima} segundos")
    print(f"Errores: {errores}")
    print(f"Tasa de error: {tasa_error}%")
    print(f"Consulta más lenta: {consulta_lenta}")


def main():
    if not os.path.exists(AGENT_LOGS):
        print("No existe logs/agent_logs.csv. Primero ejecuta agent.py.")
        return

    evaluar_precision_desde_logs()
    generar_resumen_metricas()

    print("\nArchivos generados:")
    print(RESULTS_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()