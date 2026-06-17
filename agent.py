from dotenv import load_dotenv
import os
import requests
import re

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

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


llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL"),
    temperature=0,
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_CHAT_BASE_URL"),
)

tools = [
    consultar_documentos,
    calcular_descuento,
    generar_resumen_soporte,
    planificar_atencion
]

system_prompt = """
Eres un agente funcional de atención al cliente de TechHogar S.A.

Debes razonar qué herramienta usar según la consulta del usuario.
No uses reglas fijas por palabras clave; analiza la intención de la consulta.

Herramientas disponibles:
- consultar_documentos: para garantías, stock, catálogo, devoluciones y despachos.
- calcular_descuento: para descuentos, porcentajes y precios finales.
- generar_resumen_soporte: para reclamos, problemas complejos o derivación humana.
- planificar_atencion: para organizar casos con múltiples pasos o explicar el plan de acción.

Reglas:
- Responde siempre en español.
- No inventes información.
- Si necesitas información documental, usa consultar_documentos.
- Si necesitas calcular, usa calcular_descuento.
- Si el caso requiere derivación, usa generar_resumen_soporte.
- Si el caso tiene varias etapas, usa planificar_atencion.
- Usa la memoria conversacional para mantener continuidad.
"""

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer
)


def mostrar_ultima_respuesta(resultado):
    ultimo = resultado["messages"][-1]
    return ultimo.content


def main():
    print("Agente funcional TechHogar con LangChain listo.")
    print("Escribe 'salir' para terminar.\n")

    thread_config = {
        "configurable": {
            "thread_id": "techhogar-demo"
        }
    }

    while True:
        pregunta = input("Cliente: ").strip()

        if pregunta.lower() == "salir":
            break

        resultado = agent.invoke(
            {"messages": [{"role": "user", "content": pregunta}]},
            config=thread_config
        )

        respuesta = mostrar_ultima_respuesta(resultado)

        print("\nAgente:")
        print(respuesta)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()