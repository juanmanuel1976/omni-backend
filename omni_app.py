# ==============================================================================
# OMNIQUERY - SERVIDOR DE PROTOTIPO FUNCIONAL v5.4 (Corregido y Optimizado)
# Versión final con memoria conversacional y corrección del historial.
# ==============================================================================
import asyncio
import httpx
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import AsyncAnthropic, AnthropicError

# --- CONFIGURACIÓN DE CLAVES DE API (DESDE EL ENTORNO) ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- INICIALIZACIÓN DE LA APLICACIÓN FASTAPI ---
app = FastAPI(title="OmniQuery API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELOS DE DATOS (PYDANTIC) ---
class GenerateRequest(BaseModel):
    prompt: str
    mode: str
    history: list = []

class GenerateInitialRequest(BaseModel):
    prompt: str
    history: list = []
    model: str  # 'gemini', 'deepseek', o 'claude'

class RefineRequest(BaseModel):
    prompt: str
    decisions: dict
    initial_responses: dict
    synthesis_type: str
    history: list = []

class DebateRequest(BaseModel):
    prompt: str
    history: list = []
    initial_responses: dict | None = None

# --- LÓGICA DE PROMPTS ---
def build_contextual_prompt(user_prompt: str, history: list, mode: str) -> str:
    """Construye un prompt con contexto a partir del historial de conversación."""
    history_context = ""
    if history:
        history_context += "**Historial de la Conversación Anterior (para dar contexto):**\n"
        for turn in history:
            # Asegura que las claves existen para evitar errores
            prompt_anterior = turn.get('prompt', 'N/A')
            synthesis_anterior = turn.get('synthesis', 'N/A')
            history_context += f"- **Tu Consulta Anterior:** {prompt_anterior}\n"
            history_context += f"- **Nuestra Síntesis Anterior:** {synthesis_anterior}\n---\n"

    base_prompt = ""
    if mode == 'perspectives':
        base_prompt = f"""
**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Enfoque Estratégico:** No des una respuesta directa. Tu objetivo es explorar el tema desde un ángulo único, estratégico o contraintuitivo.
**Consulta Actual del Usuario:**
"{user_prompt}"
"""
    else:  # 'direct'
        base_prompt = f"""
**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Estilo Conciso:** Sé muy breve, claro y directo en tu respuesta.
**Consulta Actual del Usuario:**
"{user_prompt}"
"""
    
    return f"{history_context}\n{base_prompt}" if history_context else base_prompt

# --- FUNCIONES AUXILIARES DE STREAMING ---
async def stream_wrapper(name, agen):
    """Envuelve un generador asíncrono para incluir el nombre del modelo en cada item."""
    async for item in agen:
        yield name, item

async def stream_merger(*agens):
    """Combina múltiples generadores asíncronos, produciendo items tan pronto como estén disponibles."""
    tasks = {asyncio.create_task(agen.__anext__()): agen for agen in agens}
    while tasks:
        done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            agen = tasks.pop(task)
            try:
                yield task.result()
                tasks[asyncio.create_task(agen.__anext__())] = agen
            except StopAsyncIteration:
                pass
            except Exception as e:
                # Si un stream falla, podemos decidir cómo manejarlo (ej. loggear, ignorar)
                print(f"Error en un stream: {e}")
                pass

# --- FUNCIONES DE STREAMING A LAS APIS ---
async def stream_gemini(prompt: str):
    if not GOOGLE_API_KEY:
        yield {"model": "gemini", "chunk": "Error: La GOOGLE_API_KEY no está configurada en el entorno."}
        return
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"model": "gemini", "chunk": f"Error HTTP {response.status_code}: {error_text.decode()}"}
                    return
                async for line in response.aiter_lines():
                    if '"text": "' in line:
                        try:
                            text_content = line.split('"text": "')[1].rsplit('"', 1)[0]
                            yield {"model": "gemini", "chunk": text_content.replace('\\n', '\n').replace('\\"', '"')}
                        except (IndexError, json.JSONDecodeError):
                            continue
    except httpx.RequestError as e:
        yield {"model": "gemini", "chunk": f"Error de conexión: {e}"}
    except Exception as e:
        yield {"model": "gemini", "chunk": f"Error inesperado en Gemini: {e}"}

async def stream_deepseek(prompt: str):
    if not DEEPSEEK_API_KEY:
        yield {"model": "deepseek", "chunk": "Error: La DEEPSEEK_API_KEY no está configurada en el entorno."}
        return
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "stream": True}
            async with client.stream("POST", "https://api.deepseek.com/chat/completions", headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"model": "deepseek", "chunk": f"Error HTTP {response.status_code}: {error_text.decode()}"}
                    return
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            content = data.get('choices', [{}])[0].get('delta', {}).get('content')
                            if content:
                                yield {"model": "deepseek", "chunk": content}
                        except (json.JSONDecodeError, IndexError):
                            continue
    except httpx.RequestError as e:
        yield {"model": "deepseek", "chunk": f"Error de conexión: {e}"}
    except Exception as e:
        yield {"model": "deepseek", "chunk": f"Error inesperado en DeepSeek: {e}"}

async def stream_claude(prompt: str):
    if not ANTHROPIC_API_KEY:
        yield {"model": "claude", "chunk": "Error: La ANTHROPIC_API_KEY no está configurada en el entorno."}
        return
    try:
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)
        async with client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                yield {"model": "claude", "chunk": text}
    except AnthropicError as e:
        yield {"model": "claude", "chunk": f"Error de API de Anthropic: {e}"}
    except Exception as e:
        yield {"model": "claude", "chunk": f"Error inesperado en Claude: {e}"}

# --- FUNCIÓN SIN STREAMING (PARA DEBATE Y SÍNTESIS) ---
async def call_ai_model_no_stream(model_name: str, prompt: str) -> str:
    """Realiza una llamada a un modelo de IA y devuelve la respuesta completa."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if model_name == "gemini":
                if not GOOGLE_API_KEY: return "Error: GOOGLE_API_KEY no configurada."
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                r = await client.post(url, json=payload)
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            elif model_name == "deepseek":
                if not DEEPSEEK_API_KEY: return "Error: DEEPSEEK_API_KEY no configurada."
                headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
                r = await client.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]

            elif model_name == "claude":
                if not ANTHROPIC_API_KEY: return "Error: ANTHROPIC_API_KEY no configurada."
                client_anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)
                msg = await client_anthropic.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                )
                return msg.content[0].text
            else:
                return f"Error: Modelo '{model_name}' no reconocido."
                
    except httpx.HTTPStatusError as e:
        return f"Error HTTP {e.response.status_code} para {model_name}: {e.response.text}"
    except Exception as e:
        return f"Error inesperado llamando a {model_name}: {e}"

# --- RUTAS DE LA APLICACIÓN ---
@app.post('/api/generate')
async def generate_initial_stream(request: GenerateRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="El campo 'prompt' no puede estar vacío.")
    
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, request.mode)
    
    async def event_stream():
        tasks = {
            "gemini": stream_gemini(contextual_prompt),
            "deepseek": stream_deepseek(contextual_prompt),
            "claude": stream_claude(contextual_prompt)
        }
        # La función stream_merger ahora está definida globalmente
        merged_agen = stream_merger(*[stream_wrapper(name, agen) for name, agen in tasks.items()])
        
        async for name, item in merged_agen:
            yield f"data: {json.dumps(item)}\n\n"
        
        yield f"data: {json.dumps({'model': 'system', 'chunk': 'DONE'})}\n\n"

    return StreamingResponse(event_stream(), media_type='text/event-stream')

@app.post('/api/generate-initial')
async def generate_initial_response(request: GenerateInitialRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="El campo 'prompt' no puede estar vacío.")
    
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct')
    
    response = await call_ai_model_no_stream(request.model, contextual_prompt)
    return {"response": response}

@app.post('/api/refine')
async def refine_and_synthesize(request: RefineRequest):
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct')
    
    active_responses = {
        k: v['content'] for k, v in request.initial_responses.items() if request.decisions.get(k) != 'discard'
    }
    highlighted_response = next(
        (v['content'] for k, v in request.initial_responses.items() if request.decisions.get(k) == 'highlight'), None
    )
    
    synthesis_prompt_parts = [f"**Consulta Original (con su historial):**\n\"{contextual_prompt}\""]
    
    if request.synthesis_type == 'summary':
        synthesis_prompt_parts.append("\n**Instrucciones:** Analiza las siguientes respuestas y crea un resumen ejecutivo que sea conciso, directo y capture los puntos clave de todas las perspectivas consideradas.")
    else: # 'report'
        synthesis_prompt_parts.append("\n**Instrucciones:** Analiza las siguientes respuestas y elabora un informe detallado y bien estructurado. Integra las diferentes perspectivas en un análisis coherente y completo.")
    
    if highlighted_response:
        synthesis_prompt_parts.append(f"\n**PERSPECTIVA DESTACADA (Dale prioridad y énfasis a esta visión en tu análisis):**\n{highlighted_response}")
    
    synthesis_prompt_parts.append("\n**RESPUESTAS A CONSIDERAR PARA LA SÍNTESIS:**")
    for model, content in active_responses.items():
        synthesis_prompt_parts.append(f"**Respuesta de {model.title()}:**\n{content}\n")
        
    final_prompt = "\n".join(synthesis_prompt_parts)
    synthesis_text = await call_ai_model_no_stream('gemini', final_prompt)
    return {"synthesis": synthesis_text}

@app.post('/api/debate')
async def debate_and_synthesize(request: DebateRequest):
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct')
    
    # Si no se proporcionan respuestas iniciales, generarlas en paralelo.
    if request.initial_responses:
        initial_responses = {k: v['content'] for k, v in request.initial_responses.items()}
    else:
        models = ['gemini', 'deepseek', 'claude']
        initial_tasks = [call_ai_model_no_stream(model, contextual_prompt) for model in models]
        initial_results = await asyncio.gather(*initial_tasks)
        initial_responses = dict(zip(models, initial_results))
    
    # Fase de crítica y refinamiento
    models_to_debate = list(initial_responses.keys())
    critique_tasks = []
    for model_to_critique in models_to_debate:
        other_responses = "\n\n".join(
            [f"**Respuesta de {m.title()}:**\n{r}" for m, r in initial_responses.items() if m != model_to_critique]
        )
        critique_prompt = (
            f"**Tu Respuesta Inicial:**\n{initial_responses[model_to_critique]}\n\n"
            f"**Respuestas de tus Colegas:**\n{other_responses}\n\n"
            "**Tu Tarea:** Actúa como un experto en el tema. Evalúa críticamente las respuestas de tus colegas. "
            "Identifica fortalezas, debilidades y puntos ciegos tanto en sus argumentos como en el tuyo. "
            "Luego, refina y fortalece tu argumento original basándote en este análisis comparativo. Tu nueva respuesta debe ser más robusta y completa."
        )
        critique_tasks.append(call_ai_model_no_stream(model_to_critique, critique_prompt))

    # **CORRECCIÓN LÓGICA:** Se asegura el orden de las respuestas
    revised_results = await asyncio.gather(*critique_tasks)
    revised_responses = dict(zip(models_to_debate, revised_results))
    
    # Fase de síntesis final
    synthesis_context = "\n\n".join(
        [f"**Argumento Revisado de {m.title()}:**\n{r}" for m, r in revised_responses.items()]
    )
    synthesis_prompt = (
        f"**Consulta Original (incluyendo historial):**\n{contextual_prompt}\n\n"
        f"**Debate de Expertos (Argumentos Revisados):**\n{synthesis_context}\n\n"
        "**Tu Tarea:** Actúa como un moderador experto y editor principal. Has recibido los argumentos finales de tres consultores de IA después de un debate. "
        "Tu objetivo es sintetizar estas perspectivas en un informe final unificado, coherente y de alto valor. "
        "Identifica el consenso, resuelve las contradicciones y presenta una conclusión clara y accionable. El resultado debe ser más que la suma de sus partes."
    )
    final_synthesis = await call_ai_model_no_stream('gemini', synthesis_prompt)

    return {
        "revised": revised_responses,
        "synthesis": final_synthesis
    }

@app.get("/health")
async def health_check():
    """Endpoint de salud para verificar que el servidor está activo."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Se recomienda obtener el puerto del entorno para mayor flexibilidad en producción
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
