# TechHogar Agent

Sistema inteligente de atención al cliente basado en modelos de lenguaje (LLM), arquitectura RAG y agentes funcionales.

El sistema permite responder consultas utilizando información contenida en documentos internos de la empresa, incorporando recuperación semántica, memoria de conversación y herramientas de razonamiento.

------------------------------------------------------------------------

## Requisitos

-   Python 3.10 o superior
-   Git
-   Token de GitHub Models con permisos de modelos

------------------------------------------------------------------------

## Configuración
1. Crear una carpeta y abrirla en visual

2.  Clonar el repositorio:

git clone https://github.com/NicolasMolinaV/techhogar.git 

3. Entrar a la carpeta cd techhogar

4.  Crear entorno virtual:

python -m venv venv

5.  Activar entorno virtual:

venv\Scripts\activate

6.  Instalar dependencias:

pip install -r requirements.txt

7.  Crear archivo `.env` en la raíz del proyecto:

GITHUB_TOKEN=tu_token 

CHAT_MODEL=openai/gpt-4o

EMBEDDING_MODEL=openai/text-embedding-3-small

GITHUB_CHAT_BASE_URL=https://models.github.ai/inference

GITHUB_EMBEDDINGS_URL=https://models.github.ai/inference/embeddings

------------------------------------------------------------------------

## Ejecución del sistema

### Paso 1: Indexar los datos

python ingest.py

Este script: - carga los documentos desde la carpeta `data/` - genera
embeddings - crea la base vectorial en `chroma_db/`

------------------------------------------------------------------------

### Paso 2: Ejecutar el asistente

python app.py

El sistema quedará en modo interactivo para ingresar consultas.

------------------------------------------------------------------------

### Paso 3: Ejecutar agente funcional

python agent.py

Este modo ejecuta el agente inteligente con:
- memoria de conversación
- recuperación semántica
- herramientas de consulta
- toma de decisiones

------------------------------------------------------------------------

### Ejemplos del agente funcional

Consulta con memoria:

Cliente:
¿Cuánto dura la garantía de Samsung?

Luego:
¿Y cuánto demora el despacho?

El agente mantiene el contexto de la conversación.

Consulta matemática:

Si un producto vale 699990 y tiene 15% de descuento, cuánto queda?

El agente utiliza automáticamente la herramienta calculadora.

------------------------------------------------------------------------

## Validación del sistema

El evaluador podrá verificar:

- Que el sistema responde correctamente
- Que las respuestas están basadas en documentos
- Que el agente utiliza herramientas según la consulta
- Que existe continuidad mediante memoria de conversación
- Que el sistema toma decisiones según el tipo de solicitud

------------------------------------------------------------------------

## Estructura del proyecto

techhogar/ ├── agent.py ├── app.py ├── ingest.py ├── requirements.txt ├── README.md
├── data/ │ ├── catalogo.txt │ ├── garantias.txt │ ├── devoluciones.txt
│ └── despachos.txt

------------------------------------------------------------------------