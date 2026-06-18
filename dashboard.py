import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = "logs/agent_logs.csv"

st.set_page_config(
    page_title="Dashboard Observabilidad TechHogar",
    layout="wide"
)

st.title("Dashboard de Observabilidad - Agente TechHogar")

st.write("""
Este dashboard permite visualizar el comportamiento del agente de IA a partir de métricas de observabilidad,
incluyendo latencia, frecuencia de errores, herramientas utilizadas y longitud de respuestas.
""")

if not os.path.exists(LOG_FILE):
    st.warning("No se encontró el archivo de logs. Ejecuta primero agent.py para generar registros.")
    st.stop()

df = pd.read_csv(LOG_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["latencia_segundos"] = pd.to_numeric(df["latencia_segundos"], errors="coerce")
df["longitud_respuesta"] = pd.to_numeric(df["longitud_respuesta"], errors="coerce")

total_consultas = len(df)
latencia_promedio = round(df["latencia_segundos"].mean(), 3)
errores = len(df[df["estado"] == "ERROR"])
tasa_error = round((errores / total_consultas) * 100, 2) if total_consultas > 0 else 0
longitud_promedio = round(df["longitud_respuesta"].mean(), 2)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total de consultas", total_consultas)
col2.metric("Latencia promedio", f"{latencia_promedio} s")
col3.metric("Errores", errores)
col4.metric("Tasa de error", f"{tasa_error}%")

st.divider()

st.subheader("Registros de ejecución")
st.dataframe(df, use_container_width=True)

st.divider()

st.subheader("Latencia por consulta")

fig1, ax1 = plt.subplots()
ax1.plot(df["timestamp"], df["latencia_segundos"], marker="o")
ax1.set_xlabel("Tiempo")
ax1.set_ylabel("Latencia en segundos")
ax1.set_title("Latencia del agente por consulta")
plt.xticks(rotation=45)
st.pyplot(fig1)

st.divider()

st.subheader("Uso de herramientas")

if "herramientas_usadas" in df.columns:
    herramientas_count = df["herramientas_usadas"].value_counts()

    fig2, ax2 = plt.subplots()
    herramientas_count.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Herramienta")
    ax2.set_ylabel("Cantidad de usos")
    ax2.set_title("Frecuencia de uso de herramientas")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
else:
    st.warning("El archivo de logs no contiene la columna herramientas_usadas.")

st.divider()

st.subheader("Estado de ejecuciones")

estado_count = df["estado"].value_counts()

fig3, ax3 = plt.subplots()
estado_count.plot(kind="bar", ax=ax3)
ax3.set_xlabel("Estado")
ax3.set_ylabel("Cantidad")
ax3.set_title("Ejecuciones exitosas vs errores")
st.pyplot(fig3)

st.divider()

st.subheader("Longitud de respuestas")

fig4, ax4 = plt.subplots()
ax4.plot(df["timestamp"], df["longitud_respuesta"], marker="o")
ax4.set_xlabel("Tiempo")
ax4.set_ylabel("Cantidad de caracteres")
ax4.set_title("Longitud de respuesta por consulta")
plt.xticks(rotation=45)
st.pyplot(fig4)

st.divider()

st.subheader("Análisis automático")

consulta_lenta = df.loc[df["latencia_segundos"].idxmax()]
respuesta_larga = df.loc[df["longitud_respuesta"].idxmax()]

st.write(f"""
- La consulta con mayor latencia fue: **{consulta_lenta['pregunta']}**
- Latencia máxima registrada: **{consulta_lenta['latencia_segundos']} segundos**
- La respuesta más extensa tuvo **{respuesta_larga['longitud_respuesta']} caracteres**
- La tasa de error registrada fue de **{tasa_error}%**
""")

if tasa_error == 0:
    st.success("No se detectaron errores críticos durante las pruebas registradas.")
else:
    st.warning("Se detectaron errores durante algunas ejecuciones. Revisar la columna error en los logs.")

if latencia_promedio > 6:
    st.warning("La latencia promedio es alta. Se recomienda optimizar recuperación RAG o reducir contexto enviado al modelo.")
else:
    st.success("La latencia promedio se mantiene dentro de un rango aceptable para el prototipo.")