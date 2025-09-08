# ==============================================================================
# OMNIQUERY - SERVIDOR DE PROTOTIPO FUNCIONAL v5.3
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
from anthropic import AsyncAnthropic

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
    # Agregamos campo opcional para recibir las respuestas iniciales desde el frontend
    initial_responses: dict = None

# --- LÓGICA DE PROMPTS ---
def build_contextual_prompt(user_prompt, history, mode):
    # Construye el historial para dar contexto a la IA
    history_context = ""
    if history:
        history_context += "**Historial de la Conversación Anterior (para dar contexto):**\n"
        for turn in history:
            history_context += f"- **Tu Consulta Anterior:** {turn.get('prompt', 'N/A')}\n"
            history_context += f"- **Nuestra Síntesis Anterior:** {turn.get('synthesis', 'N/A')}\n---\n"

    base_prompt = ""
    if mode == 'perspectives':
        base_prompt = f"""
**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Enfoque Estratégico:** No des una respuesta directa. Tu objetivo es explorar el tema desde un ángulo único o estratégico.
**Consulta Actual del Usuario:**
"{user_prompt}"
"""
    else: # direct
        base_prompt = f"""
**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Estilo Conciso:** Sé muy breve y directo.
**Consulta Actual del Usuario:**
"{user_prompt}"
"""
    
    if history_context:
        return f"{history_context}\n{base_prompt}"
    return base_prompt

# --- FUNCIONES DE STREAMING ---
async def stream_gemini(prompt):
    if not GOOGLE_API_KEY: 
        yield {"model": "gemini", "chunk": "Error: GOOGLE_API_KEY no configurada."}
        return
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"model": "gemini", "chunk": f"Error: {error_text.decode()}"}
                    return
                async for line in response.aiter_lines():
                    if '"text": "' in line:
                        try:
                            text_content = line.split('"text": "')[1].rsplit('"', 1)[0]
                            yield {"model": "gemini", "chunk": text_content.replace('\\n', '\n').replace('\\"', '"')}
                        except IndexError: 
                            continue
    except Exception as e: 
        yield {"model": "gemini", "chunk": f"Error: {e}"}

async def stream_deepseek(prompt):
    if not DEEPSEEK_API_KEY: 
        yield {"model": "deepseek", "chunk": "Error: DEEPSEEK_API_KEY no configurada."}
        return
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "stream": True}
            async with client.stream("POST", "https://api.deepseek.com/chat/completions", headers=headers, json=payload)极 response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"model": "deepseek", "chunk": f"Error: {error_text.decode()}"}
                    return
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[极DONE]': 
                            break
                        try:
                            data = json.loads(data_str)
                            if data.get('choices', [{}])[0].get('delta', {}).get('content'):
                                yield {"model": "deepseek", "chunk": data['choices'][0]['delta']['content']}
                        except json.JSONDecodeError: 
                            continue
    except Exception as e: 
        yield {"model": "deepseek", "chunk": f"Error: {e}"}

async def stream_claude(prompt):
    if not ANTHROPIC_API_KEY: 
        yield {"model": "claude", "chunk": "Error: ANTHROPIC_API_KEY no configurada."}
        return
    try:
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)
        async with client.messages.stream(model="claude-3-haiku-20240307", max_tokens=4096, messages=[{"role": "user", "content": prompt}]) as stream:
            async for text in stream.text_stream:
                yield {"model": "claude", "chunk": text}
    except Exception as e: 
        yield {"model": "claude", "chunk": f"Error: {e}"}


# --- FUNCIONES SIN STREAMING (PARA DEBATE Y SÍNTESIS) ---
async def call_ai_model_no_stream(model_name: str, prompt: str):
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if model_name == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
                payload = {"contents": [{"parts极[{"text": prompt}]}]}
                r = await client.post(url, json=payload)
                if r.status_code != 200: 
                    return f"Error HTTP {r.status_code}: {r.text}"
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif model_name == "deepseek":
                headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
                payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
                r = await client.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload)
                if r.status_code != 200: 
                    return f"Error HTTP {r.status_code}: {极r.text}"
                return r.json()["choices"][0]["message"]["content"]
            elif model_name == "claude":
                client_anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)
                msg = await client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=4096, messages=[{"role": "user", "content": prompt}])
                return msg.content[0].text
    except Exception as e: 
        return f"Error: {e}"

# --- RUTAS DE LA APLICACIÓN ---
@app.post('/api/generate')
async def generate_initial_stream(request: GenerateRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    # CORRECCIÓN: Se construye el prompt con el historial
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, request.mode)
    
    async def event_stream():
        tasks = { 
            "gemini": stream_gemini(contextual_prompt), 
            "deepseek": stream_deepseek(contextual_prompt), 
            "claude": stream_claude(contextual_prompt) 
        }
        async def stream_wrapper(name, agen):
            async for item in agen: 
                yield name, item
        merged_agen = stream_merger(*[stream_wrapper(name, agen) for name, agen in tasks.items()])
        async for name, item in merged_agen:
             yield f"data: {json.dumps(item)}\n\n"
        yield f"data: {json.dumps({'model': 'system', 'chunk': 'DONE'})}\n\n"

    async def stream_merger(*agens):
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

    return StreamingResponse(event_stream(), media_type='text/event-stream')

# Nuevo endpoint para generar una respuesta inicial de un modelo específico
@app.post('/api/generate-initial')
async def generate_initial_response(request: GenerateInitialRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct')
    
    response = await call_ai_model_no_stream(request.model, contextual_prompt)
    return {"response": response}


@app.post('/api/refine')
async def refine_and_synthesize(request: RefineRequest):
    # CORRECCIÓN: Se construye el prompt con el historial
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct')
    
    active_responses = {k: v['content'] for k, v in request.initial_responses.items() if request.decisions.get(k) != 'discard'}
    highlighted_response = next((v['content'] for k, v in request.initial_responses.items() if request.decisions.get(k) == 'highlight'), None)
    
    synthesis_prompt_parts = [f"**Consulta Original (con su historial):**\n\"{contextual_prompt}\""]
    if request.synthesis_type == 'summary':
        synthesis_prompt_parts.append("\n**Instrucciones:** Crea un resumen ejecutivo, conciso y directo.")
    else:
        synthesis_prompt_parts.append("\n**Instrucciones:** Elabora un informe detallado y bien estructurado.")
    if highlighted_response:
        synthesis_prompt_parts.append("\n**PERSPECTIVA DESTACADA (Priorizar):**\n" + highlighted_response)
    synthesis_prompt_parts.append("\n极**RESPUESTAS A CONSIDERAR:**")
    for model, content in active_responses.items():
        synthesis_prompt_parts.append(f"**Respuesta de {model.title()}:**\n{content}\n")
    final_prompt = "\n".join(synthesis_prompt_parts)
    synthesis_text = await call_ai_model_no_stream('gemini', final_prompt)
    return {"synthesis": synthesis_text}

@app.post('/api/debate')
async def debate_and_synthesize(request: DebateRequest):
    # CORRECCIÓN: Se construye el prompt con el historial
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct')
    
    # Si se proporcionaron respuestas iniciales, usarlas. De lo contrario, generarlas.
    if request.initial_responses is not None:
        initial_responses = request.initial_responses
    else:
        # Obtener respuestas iniciales de cada modelo
        initial_tasks = [
            call_ai_model_no_stream('gemini', contextual_prompt),
            call_ai_model_no_stream('deepseek', contextual_prompt),
            call_ai_model_no_stream('claude', contextual_prompt)
        ]
        initial_results = await asyncio.gather(*initial_tasks)
        initial_responses = {
            'gemini': initial_results[0],
            'deepseek': initial_results[1],
            'claude': initial_results[2]
        }
    
    # Fase de crítica y refinamiento
    critique_prompts = {}
    for model in initial_responses:
        context = "\n\n".join([f"**Respuesta de {m.title()}:**\n{r}" for m, r in initial_responses.items() if m != model])
        critique_prompts[model] = f"**Tu Respuesta Inicial:**\n{initial_responses[model]}\n\n**Respuestas de Colegas:**\n{context}\n\n**Tu Tarea:** Critica sus respuestas y refina tu argumento."
    
    critique_tasks = [call_ai_model_no_stream(model, prompt) for model, prompt in critique_prompts.items()]
    revised_results = await asyncio.gather(*critique_tasks)
    revised_responses = {
        'gemini': revised_results[0],
        'deepseek': revised_results[1],
        'claude': revised_results[2]
    }
    
    synthesis_context = "\n\n".join([f"**Argumento Revisado de {m.title()}:**\n{r}" for m, r in revised_responses.items()])
    synthesis_prompt = f"**Consulta (con historial):**\n{contextual_prompt}\n\n**Debate de Expertos:**\n{synthesis_context}\n\n**Tu Tarea:** Modera y crea un informe final unificado."
    final_synthesis = await call_ai_model_no_stream('gemini', synthesis_prompt)

    return {
        "revised": revised_responses, 
        "synthesis": final_synthesis
    }

# Para evitar que el servidor se cierre por inactividad, podemos agregar un endpoint de salud
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
