# ==============================================================================
# OMNIQUERY - SERVIDOR FASTAPI v5.1 (con Síntesis Final)
# Versión nativa ASGI compatible con uvicorn y con lógica de refinamiento.
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

# --- CONFIGURACIÓN DE CLAVES DE API ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- INICIALIZACIÓN DE LA APLICACIÓN FASTAPI ---
app = FastAPI(title="OmniQuery API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELOS PYDANTIC ---
class GenerateRequest(BaseModel):
    prompt: str

class RefineRequest(BaseModel):
    prompt: str
    decisions: dict
    initial_responses: dict

# --- PROMPT INICIAL MEJORADO ---
def get_initial_prompt(user_prompt):
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
        
# --- SÍNTESIS FINAL ---
def build_refine_prompt(original_prompt, highlighted, normal):
    """Construye el prompt para la síntesis basado en las decisiones del usuario."""
    
    prompt_parts = [
        "**ROL Y OBJETIVO:** Eres un analista experto encargado de sintetizar múltiples perspectivas de IA en una única respuesta superior, coherente y bien estructurada. Tu respuesta debe estar exclusivamente en español.",
        "",
        f"**CONSULTA ORIGINAL DEL USUARIO:**\n\"{original_prompt}\"",
        ""
    ]
    
    if highlighted:
        prompt_parts.append("**PERSPECTIVA DESTACADA (Fuente principal para tu análisis):**")
        for model, content in highlighted.items():
            prompt_parts.extend([f"**Respuesta de {model.title()}:**\n{content}", ""])
    
    if normal:
        prompt_parts.append("**PERSPECTIVAS ADICIONALES (Contexto secundario para enriquecer la respuesta):**")
        for model, content in normal.items():
            prompt_parts.extend([f"**Respuesta de {model.title()}:**\n{content}", ""])
    
    prompt_parts.extend([
        "**TAREA FINAL:**",
        "Basándote en la consulta original y, sobre todo, en la **perspectiva destacada**, genera un informe final unificado.",
        "Tu informe debe:",
        "- Integrar los puntos más fuertes de la respuesta destacada de forma fluida.",
        "- Usar las perspectivas adicionales solo para añadir detalles o matices que enriquezcan la idea central, sin contradecirla.",
        "- Ser más completo, claro y refinado que cualquiera de las respuestas individuales.",
        "- Mantener un tono profesional y directo.",
        "\n**COMIENZA TU SÍNTESIS FINAL AQUÍ:**"
    ])
    
    return "\n".join(prompt_parts)

async def generate_synthesis_with_gemini(prompt):
    """Genera la síntesis usando Gemini (sin streaming)."""
    if not GOOGLE_API_KEY:
        raise Exception("GOOGLE_API_KEY no está configurada")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        response = await client.post(url, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error de Gemini: {response.status_code} - {response.text}")
        
        result = response.json()
        
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise Exception("Formato de respuesta inesperado de Gemini")


# --- RUTAS DE LA APLICACIÓN ---
@app.post('/api/generate')
async def generate_initial_stream(request: GenerateRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    initial_prompt = get_initial_prompt(request.prompt)
    
    async def event_stream():
        tasks = {
            "gemini": stream_gemini(initial_prompt),
            "deepseek": stream_deepseek(initial_prompt),
            "claude": stream_claude(initial_prompt)
        }
        pending_tasks = [asyncio.create_task(tasks[name].__anext__(), name=name) for name in tasks]
        
        while pending_tasks:
            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                model_name = task.get_name()
                try:
                    result = task.result()
                    yield f"data: {json.dumps(result)}\n\n"
                    new_task = asyncio.create_task(tasks[model_name].__anext__(), name=model_name)
                    pending_tasks.add(new_task)
                except StopAsyncIteration:
                    continue
        
        yield f"data: {json.dumps({'model': 'system', 'chunk': 'DONE'})}\n\n"
    
    return StreamingResponse(event_stream(), media_type='text/event-stream')

@app.post('/api/refine')
async def refine_and_synthesize(request: RefineRequest):
    highlighted = {m: c['content'] for m, c in request.initial_responses.items() if request.decisions.get(m) == 'highlight'}
    normal = {m: c['content'] for m, c in request.initial_responses.items() if request.decisions.get(m) == 'normal'}

    synthesis_prompt = build_refine_prompt(request.prompt, highlighted, normal)
    
    try:
        final_synthesis = await generate_synthesis_with_gemini(synthesis_prompt)
        return {"synthesis": final_synthesis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "OmniQuery API v5.1 - FastAPI con Síntesis Final"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
