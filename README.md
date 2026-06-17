# TechHogar Agent

Sistema inteligente de atención al cliente basado en modelos de lenguaje (LLM), arquitectura RAG y agentes funcionales.

El sistema permite responder consultas utilizando información contenida en documentos internos de la empresa, incorporando recuperación semántica, memoria de conversación, herramientas de razonamiento, escritura y planificación.

La versión de agente funcional fue implementada con LangChain Agents mediante `create_agent`, permitiendo que el modelo de lenguaje participe en la selección de herramientas disponibles. A diferencia de un enrutador por palabras clave, el LLM analiza la intención de la consulta y decide qué herramienta utilizar dentro del flujo de trabajo.

------------------------------------------------------------------------

## Requisitos

- Python 3.10 o superior
- Git
- Token de GitHub Models con permisos de modelos

------------------------------------------------------------------------

## Configuración

1. Crear una carpeta y abrirla en Visual Studio Code.

2. Clonar el repositorio:

git clone https://github.com/NicolasMolinaV/techhogar.git

3. Entrar a la carpeta del proyecto:

cd techhogar

4. Crear entorno virtual:

python -m venv venv

5. Activar entorno virtual:

venv\Scripts\activate

6. Instalar dependencias:

pip install -r requirements.txt

7. Crear archivo `.env` en la raíz del proyecto:

GITHUB_TOKEN=tu_token

CHAT_MODEL=openai/gpt-4o

EMBEDDING_MODEL=openai/text-embedding-3-small

GITHUB_CHAT_BASE_URL=https://models.github.ai/inference

GITHUB_EMBEDDINGS_URL=https://models.github.ai/inference/embeddings

------------------------------------------------------------------------

## Ejecución del sistema

### Paso 1: Indexar los datos

python ingest.py

Este script:

- carga los documentos desde la carpeta `data/`
- genera embeddings mediante GitHub Models
- crea la base vectorial en `chroma_db/`

------------------------------------------------------------------------

### Paso 2: Ejecutar el asistente RAG básico

python app.py

Este modo ejecuta el sistema RAG base, permitiendo realizar consultas sobre los documentos internos de TechHogar.

------------------------------------------------------------------------

### Paso 3: Ejecutar el agente funcional

python agent.py

Este modo ejecuta el agente inteligente con:

- LangChain Agents mediante `create_agent`
- memoria de corto plazo mediante `InMemorySaver`
- recuperación semántica con ChromaDB
- herramientas de consulta, cálculo, escritura y planificación
- toma de decisiones realizada por el modelo de lenguaje

------------------------------------------------------------------------

## Herramientas del agente

El agente utiliza distintas herramientas dependiendo del tipo de consulta realizada por el usuario. Estas herramientas son entregadas al agente mediante LangChain, permitiendo que el LLM analice la intención de la consulta y seleccione cuál utilizar.

| Herramienta | Función |
|---|---|
| `consultar_documentos` | Busca información dentro de los documentos de la empresa usando RAG y ChromaDB |
| `calcular_descuento` | Realiza cálculos matemáticos, descuentos y precios finales |
| `generar_resumen_soporte` | Genera resúmenes para derivación a soporte humano |
| `planificar_atencion` | Organiza casos con múltiples pasos antes de responder |

------------------------------------------------------------------------

## Memoria del agente

El agente incorpora memoria de corto plazo mediante `InMemorySaver`, permitiendo mantener continuidad dentro de una misma conversación.

La memoria se gestiona usando un `thread_id`, lo que permite que el agente conserve el historial de interacción y pueda responder consultas posteriores considerando el contexto anterior.

------------------------------------------------------------------------

## Ejemplos del agente funcional

### Consulta documental

Cliente:

¿Cuánto dura la garantía de Samsung?

Respuesta esperada:

La garantía de los productos Samsung ofrecidos por TechHogar tiene una duración mínima de 12 meses.

------------------------------------------------------------------------

### Consulta con memoria

Cliente:

¿Cuánto dura la garantía de Samsung?

Luego:

¿Y qué pasa si necesito despacho?

El agente mantiene el contexto de la conversación y responde utilizando la información de despacho disponible en los documentos.

------------------------------------------------------------------------

### Consulta matemática

Cliente:

Si un producto vale 699990 y tiene 15% de descuento, cuánto queda?

Respuesta esperada:

El precio final del producto, aplicando un descuento del 15%, queda en $594.992 aproximadamente.

------------------------------------------------------------------------

### Consulta de derivación

Cliente:

Tengo un reclamo y necesito ayuda humana

Respuesta esperada:

El agente genera un resumen de la solicitud y deriva el caso a soporte humano especializado.

------------------------------------------------------------------------

### Consulta con planificación

Cliente:

Necesito resolver un caso donde quiero saber garantía, despacho y posible derivación

El agente organiza la solicitud en distintas etapas, consulta información documental, entrega la respuesta correspondiente y considera la posibilidad de derivación.

------------------------------------------------------------------------

## Validación del sistema

El evaluador podrá verificar:

- que el sistema RAG base indexa documentos correctamente
- que se generan embeddings mediante GitHub Models
- que ChromaDB recupera contexto relevante
- que el agente funcional utiliza LangChain Agents mediante `create_agent`
- que el LLM participa en la selección de herramientas
- que existe memoria conversacional mediante `InMemorySaver`
- que se integran herramientas de consulta, cálculo, escritura y planificación
- que el agente responde de forma coherente según el tipo de solicitud

------------------------------------------------------------------------

## Estructura del proyecto

techhogar/
├── agent.py
├── app.py
├── ingest.py
├── requirements.txt
├── README.md
├── data/
│   ├── catalogo.txt
│   ├── garantias.txt
│   ├── devoluciones.txt
│   └── despachos.txt
├── docs/
│   └── diagrama_orquestacion_agente.png

------------------------------------------------------------------------

## Diagrama de orquestación

El repositorio incluye un diagrama de orquestación en la carpeta `docs/`, donde se representa la relación entre el usuario, el agente LangChain, el modelo de lenguaje, las herramientas, la memoria y la base vectorial.

Archivo:

docs/diagrama_orquestacion_agente.png

------------------------------------------------------------------------

## Decisiones de diseño

| Componente | Decisión | Justificación |
|---|---|---|
| LangChain Agent | Uso de `create_agent` | Permite que el LLM seleccione herramientas dentro del flujo del agente |
| ChromaDB | Base vectorial | Permite recuperación semántica sobre documentos internos |
| GitHub Models | LLM y embeddings | Permite usar modelos mediante token y endpoints compatibles |
| InMemorySaver | Memoria de corto plazo | Mantiene continuidad conversacional durante la sesión |
| RAG | Consulta documental | Reduce respuestas inventadas al responder con contexto recuperado |
| Herramientas | Consulta, cálculo, escritura y planificación | Permiten cubrir distintos tipos de tareas organizacionales |

------------------------------------------------------------------------

## Notas

- El archivo `.env` no debe subirse a GitHub.
- Los datos utilizados son simulados para fines académicos.
- El sistema corresponde a un prototipo funcional de agente organizacional basado en IA.
- Antes de ejecutar `agent.py`, se debe ejecutar `python ingest.py` para crear la base vectorial.