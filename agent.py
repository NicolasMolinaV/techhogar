import os
import requests
import re
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from observability import registrar_ejecucion

load_dotenv(override=True)

PERSIST_DIR = "chroma_db"

class GitHubEmbeddings:
    def __init__(self, model: str, token: str, url: str):
        self.model = model
        self.token = token
        self.url = url

    def _embed(self, texts):
        clean_texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not clean_texts:
            return []

        response = requests.post(
            self.url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": clean_texts,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]

def crear_retriever():
    embeddings = GitHubEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        token=os.getenv("GITHUB_TOKEN"),
        url=os.getenv("GITHUB_EMBEDDINGS_URL"),
    )
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = crear_retriever()

@tool
def consultar_documentos(pregunta: str) -> str:
    """
    Consulta documentos internos de TechHogar.
    Útil para preguntas sobre garantías, stock, catálogo, devoluciones y despachos.
    """
    docs = retriever.invoke(pregunta)
    if not docs:
        return "No se encontró información suficiente en los documentos."
    contexto = []
    for doc in docs:
        fuente = doc.metadata.get("source", "fuente desconocida")
        contexto.append(f"Fuente: {fuente}\nContenido: {doc.page_content}")
    return "\n\n".join(contexto)

@tool
def calcular_descuento(consulta: str) -> str:
    """
    Calcula descuentos, precios finales y operaciones matemáticas simples.
    Útil cuando el cliente pregunta por porcentajes, descuentos o valores finales.
    """
    numeros = re.findall(r"\d+", consulta)
    if len(numeros) >= 2 and "%" in consulta:
        precio = float(numeros[0])
        descuento = float(numeros[1])
        resultado = precio - (precio * descuento / 100)
        return f"El precio final con {descuento}% de descuento es {round(resultado)}."
    return "No se pudo identificar una operación matemática válida."

@tool
def generar_resumen_soporte(consulta: str) -> str:
    """
    Genera un resumen escrito para derivar un caso a soporte humano.
    Útil para reclamos, problemas complejos o solicitudes de ayuda humana.
    """
    return f"""
Resumen para soporte humano:
Solicitud del cliente:
{consulta}
Recomendación:
Derivar el caso a un agente especializado para revisión.
"""

@tool
def planificar_atencion(consulta: str) -> str:
    """
    Genera un plan breve de atención para consultas con múltiples pasos.
    Útil cuando el caso requiere ordenar acciones antes de responder.
    """
    return f"""
Plan de atención:
1. Identificar la intención principal del cliente.
2. Determinar si requiere consulta documental, cálculo o derivación.
3. Usar la herramienta correspondiente.
4. Generar una respuesta clara y basada en evidencia.
5. Mantener continuidad usando memoria conversacional si existen mensajes previos.

Consulta analizada:
{consulta}
"""

class MemoriaAgente:
    """
    Clase que implementa una estrategia de memoria para el agente.
    Actualmente envuelve el InMemorySaver de LangGraph, pero puede ser extendida
    con estrategias de limpieza, resumen o persistencia en base de datos.
    """
    def __init__(self):
        self.checkpointer = InMemorySaver()

    def obtener_checkpointer(self):
        return self.checkpointer

class AgenteTechHogar:
    """
    Clase que encapsula el agente de razonamiento de TechHogar.
    Utiliza LangGraph y un enfoque ReAct para planificar y decidir qué herramientas ejecutar.
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("CHAT_MODEL"),
            temperature=0,
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url=os.getenv("GITHUB_CHAT_BASE_URL"),
        )
        
        self.tools = [
            consultar_documentos,
            calcular_descuento,
            generar_resumen_soporte,
            planificar_atencion
        ]
        
        self.memoria = MemoriaAgente()
        
        self.system_prompt = """Eres un agente funcional de atención al cliente de TechHogar S.A.

Debes razonar paso a paso qué herramienta usar según la consulta del usuario.
Eres un agente Reactivo (ReAct). No sigues reglas fijas por palabras clave; en su lugar, analizas la intención de la consulta y decides autónomamente qué herramienta ejecutar. Si necesitas planificar un caso complejo, puedes hacerlo explícitamente en tu pensamiento o usando la herramienta de planificación.

Herramientas disponibles:
- consultar_documentos: obligatoria si te preguntan sobre garantías, stock, catálogo, devoluciones y despachos. Nunca asumas información de esto sin consultar.
- calcular_descuento: para calcular descuentos, porcentajes y precios finales.
- generar_resumen_soporte: para derivar reclamos, problemas complejos o solicitudes que escapen de tus herramientas a un humano.
- planificar_atencion: para organizar casos con múltiples pasos.

Reglas:
- Responde siempre en español.
- No inventes información.
- Basa tu respuesta en el uso de herramientas, ejecutando una a la vez y evaluando el resultado.
- Usa la memoria conversacional para mantener la continuidad si hay mensajes previos."""

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
            checkpointer=self.memoria.obtener_checkpointer()
        )
        
    def invocar(self, pregunta: str, session_id: str = "techhogar-demo"):
        thread_config = {"configurable": {"thread_id": session_id}}
        return self.agent.invoke(
            {"messages": [("user", pregunta)]},
            config=thread_config
        )

def mostrar_ultima_respuesta(resultado):
    ultimo = resultado["messages"][-1]
    return ultimo.content

def obtener_herramientas_usadas(resultado):
    herramientas = []
    for mensaje in resultado.get("messages", []):
        if hasattr(mensaje, "tool_calls") and mensaje.tool_calls:
            for llamada in mensaje.tool_calls:
                nombre = llamada.get("name", "herramienta_desconocida")
                herramientas.append(nombre)
    if not herramientas:
        return "sin_tool_detectada"
    return ", ".join(sorted(set(herramientas)))

def main():
    print("Inicializando Agente Funcional TechHogar (LangGraph ReAct)...")
    try:
        agente_app = AgenteTechHogar()
    except Exception as e:
        print(f"Error al inicializar el agente: {e}")
        return

    print("Agente funcional TechHogar con observabilidad listo.")
    print("Escribe 'salir' para terminar.\n")

    session_id = "techhogar-demo"

    while True:
        try:
            pregunta = input("Cliente: ").strip()
        except EOFError:
            break
            
        if not pregunta:
            continue
            
        if pregunta.lower() == "salir":
            break

        inicio = time.perf_counter()

        try:
            resultado = agente_app.invocar(pregunta, session_id)
            fin = time.perf_counter()
            latencia = fin - inicio

            respuesta = mostrar_ultima_respuesta(resultado)
            herramientas_usadas = obtener_herramientas_usadas(resultado)

            registrar_ejecucion(
                pregunta=pregunta,
                respuesta=respuesta,
                latencia=latencia,
                herramientas_usadas=herramientas_usadas,
                estado="OK",
                error=""
            )

            print("\nHerramientas usadas:")
            print(herramientas_usadas)

            print("\nLatencia:")
            print(f"{round(latencia, 3)} segundos")

            print("\nAgente:")
            print(respuesta)
            print("\n" + "-" * 50 + "\n")

        except Exception as e:
            fin = time.perf_counter()
            latencia = fin - inicio
            error = str(e)

            registrar_ejecucion(
                pregunta=pregunta,
                respuesta="",
                latencia=latencia,
                herramientas_usadas="error",
                estado="ERROR",
                error=error
            )

            print("\nOcurrió un error durante la ejecución del agente:")
            print(error)
            print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()