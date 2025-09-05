# ==============================================================================
# OMNIQUERY - SERVIDOR DE PROTOTIPO FUNCIONAL v5.2
# Versión final con todos los modos, incluyendo /api/debate
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

class RefineRequest(BaseModel):
    prompt: str
    decisions: dict
    initial_responses: dict
    synthesis_type: str

class DebateRequest(BaseModel):
    prompt: str
    history: list = []

# --- LÓGICA DE PROMPTS ---
def get_initial_prompt(user_prompt, mode):
    if mode == 'perspectives':
        return f"""
**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Enfoque Estratégico:** No des una respuesta directa. Tu objetivo es explorar el tema desde un ángulo único o estratégico. Analiza el 'porqué' detrás de la pregunta, los supuestos implícitos o las consecuencias a largo plazo. Ofrece una perspectiva que invite a la reflexión.
**Consulta del Usuario:**
"{user_prompt}"
"""
    return f"""
**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Estilo Conciso:** Para esta primera respuesta, sé muy breve y directo. Ofrece un resumen ejecutivo, los puntos clave o una respuesta inicial clara. Evita introducciones largas y formalidades. El objetivo es dar una primera impresión rápida y útil.
**Consulta del Usuario:**
"{user_prompt}"
"""

# --- FUNCIONES DE STREAMING ---
async def stream_gemini(prompt):
    if not GOOGLE_API_KEY:
        yield {"model": "gemini", "chunk": "Error: GOOGLE_API_KEY no está configurada."}
        return
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"model": "gemini", "chunk": f"Error Gemini: {error_text.decode()}"}
                    return
                async for line in response.aiter_lines():
                    if '"text": "' in line:
                        try:
                            text_content = line.split('"text": "')[1].rsplit('"', 1)[0]
                            clean_text = text_content.replace('\\n', '\n').replace('\\"', '"')
                            yield {"model": "gemini", "chunk": clean_text}
                        except IndexError:
                            continue
    except Exception as e:
        yield {"model": "gemini", "chunk": f"Error en stream Gemini: {e}"}

async def stream_deepseek(prompt):
    if not DEEPSEEK_API_KEY:
        yield {"model": "deepseek", "chunk": "Error: DEEPSEEK_API_KEY no está configurada."}
        return
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "stream": True}
            async with client.stream("POST", "https://api.deepseek.com/chat/completions", headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"model": "deepseek", "chunk": f"Error DeepSeek: {error_text.decode()}"}
                    return
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]': break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and data['choices'][0]['delta'].get('content'):
                                yield {"model": "deepseek", "chunk": data['choices'][0]['delta']['content']}
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield {"model": "deepseek", "chunk": f"Error en stream DeepSeek: {e}"}

async def stream_claude(prompt):
    if not ANTHROPIC_API_KEY:
        yield {"model": "claude", "chunk": "Error: ANTHROPIC_API_KEY no está configurada."}
        return
    try:
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)
        async with client.messages.stream(model="claude-3-haiku-20240307", max_tokens=4096, messages=[{"role": "user", "content": prompt}]) as stream:
            async for text in stream.text_stream:
                yield {"model": "claude", "chunk": text}
    except Exception as e:
        yield {"model": "claude", "chunk": f"Error en stream Claude: {e}"}

# --- FUNCIONES SIN STREAMING (PARA DEBATE Y SÍNTESIS) ---
async def call_ai_model_no_stream(model_name: str, prompt: str):
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if model_name == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                r = await client.post(url, json=payload)
                if r.status_code != 200: return f"Error HTTP {r.status_code}: {r.text}"
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            elif model_name == "deepseek":
                headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
                r = await client.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload)
                if r.status_code != 200: return f"Error HTTP {r.status_code}: {r.text}"
                return r.json()["choices"][0]["message"]["content"]

            elif model_name == "claude":
                client_anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)
                msg = await client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=4096, messages=[{"role": "user", "content": prompt}])
                return msg.content[0].text

    except Exception as e: return f"Error Detallado: {type(e).__name__} - {e}"

# --- RUTAS DE LA APLICACIÓN ---
@app.post('/api/generate')
async def generate_initial_stream(request: GenerateRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    initial_prompt = get_initial_prompt(request.prompt, request.mode)
    
    async def event_stream():
        tasks = {
            "gemini": stream_gemini(initial_prompt),
            "deepseek": stream_deepseek(initial_prompt),
            "claude": stream_claude(initial_prompt)
        }
        
        async def stream_wrapper(name, agen):
            async for item in agen:
                yield name, item

        merged_agen = stream_merger(*[stream_wrapper(name, agen) for name, agen in tasks.items()])

        async for name, item in merged_agen:
             yield f"data: {json.dumps(item)}\n\n"

        yield f"data: {json.dumps({'model': 'system', 'chunk': 'DONE'})}\n\n"

    # Helper para mezclar generadores asíncronos
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

@app.post('/api/refine')
async def refine_and_synthesize(request: RefineRequest):
    active_responses = {k: v['content'] for k, v in request.initial_responses.items() if request.decisions.get(k) != 'discard'}
    highlighted_response = next((v['content'] for k, v in request.initial_responses.items() if request.decisions.get(k) == 'highlight'), None)
    
    synthesis_prompt_parts = [
        f"**Consulta Original del Usuario:**\n\"{request.prompt}\"",
        "\n**Instrucciones para la Síntesis Final:**"
    ]
    if request.synthesis_type == 'summary':
        synthesis_prompt_parts.append("Crea un resumen ejecutivo, conciso y directo que combine los puntos clave.")
    else: # report
        synthesis_prompt_parts.append("Elabora un informe detallado y bien estructurado. Sintetiza los mejores insights, añade profundidad y ofrece una conclusión comprehensiva.")

    if highlighted_response:
        synthesis_prompt_parts.append("\n**PERSPECTIVA DESTACADA (Priorizar este enfoque):**")
        synthesis_prompt_parts.append(highlighted_response)

    synthesis_prompt_parts.append("\n**RESPUESTAS A CONSIDERAR:**")
    for model, content in active_responses.items():
        synthesis_prompt_parts.append(f"**Respuesta de {model.title()}:**\n{content}\n")
    
    final_prompt = "\n".join(synthesis_prompt_parts)
    synthesis_text = await call_ai_model_no_stream('gemini', final_prompt)
    
    return {"synthesis": synthesis_text}

# --- NUEVO ENDPOINT PARA EL MODO DEBATE ---
@app.post('/api/debate')
async def debate_and_synthesize(request: DebateRequest):
    # Paso 1: Obtener respuestas iniciales
    initial_prompt = get_initial_prompt(request.prompt, 'direct') # El debate empieza con respuestas directas
    initial_tasks = [call_ai_model_no_stream(model, initial_prompt) for model in ['gemini', 'deepseek', 'claude']]
    initial_results = await asyncio.gather(*initial_tasks)
    initial_responses = {'gemini': initial_results[0], 'deepseek': initial_results[1], 'claude': initial_results[2]}

    # Paso 2: Crítica cruzada
    critique_prompts = {}
    for model_to_critique in initial_responses:
        context = "\n\n".join([f"**Respuesta de {m.title()}:**\n{r}" for m, r in initial_responses.items() if m != model_to_critique])
        critique_prompts[model_to_critique] = f"""
**Tu Respuesta Inicial:**\n{initial_responses[model_to_critique]}

**Respuestas de tus Colegas:**\n{context}

**Tu Tarea:**
Actúa como un experto crítico. Revisa las respuestas de tus colegas.
1.  **Identifica Debilidades:** Encuentra al menos una debilidad, suposición incorrecta u omisión en sus argumentos.
2.  **Refuerza tu Postura:** Basado en sus respuestas, refuerza o revisa tu argumento inicial para hacerlo más robusto.
3.  **Formato:** Responde de forma concisa con tu análisis revisado.
"""
    
    critique_tasks = [call_ai_model_no_stream(model, prompt) for model, prompt in critique_prompts.items()]
    revised_results = await asyncio.gather(*critique_tasks)
    revised_responses = {'gemini': revised_results[0], 'deepseek': revised_results[1], 'claude': revised_results[2]}

    # Paso 3: Síntesis final
    synthesis_context = "\n\n".join([f"**Argumento Revisado de {m.title()}:**\n{r}" for m, r in revised_responses.items()])
    synthesis_prompt = f"""
**Consulta Original:**\n{request.prompt}

**Debate de Expertos:**
Tres IAs han debatido y revisado sus posturas. Estos son sus argumentos finales:
{synthesis_context}

**Tu Tarea como Moderador:**
Crea un informe final y unificado. Sintetiza los puntos de acuerdo, aclara las discrepancias y ofrece una conclusión ejecutiva basada en el consenso alcanzado en el debate.
"""
    final_synthesis = await call_ai_model_no_stream('gemini', synthesis_prompt)

    return {"revised": revised_responses, "synthesis": final_synthesis}
