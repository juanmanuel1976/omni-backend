# ==============================================================================
# OMNIQUERY - SERVIDOR FASTAPI v8.0 (con Memoria Conversacional)
# ==============================================================================
import asyncio
import httpx
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Dict

from anthropic import AsyncAnthropic

# --- CONFIGURACIÓN DE CLAVES DE API ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- INICIALIZACIÓN DE LA APLICACIÓN FASTAPI ---
app = FastAPI(title="OmniQuery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- MODELOS PYDANTIC ---
# MODIFICADO: GenerateRequest ahora acepta un historial opcional
class GenerateRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []

class RefineRequest(BaseModel):
    prompt: str
    decisions: dict
    initial_responses: dict
    synthesis_type: Literal['summary', 'report'] = 'summary'

# --- SISTEMA DE PROMPTS CON MEMORIA ---

def build_history_context(history: List[Dict[str, str]]) -> str:
    """Construye una cadena de texto formateada a partir del historial."""
    if not history:
        return ""
    
    context = ["**Historial de la Conversación Anterior (Contexto Clave):**"]
    for i, turn in enumerate(history):
        context.append(f"**Turno {i+1} - Consulta del Usuario:**\n{turn.get('prompt', '')}")
        context.append(f"**Turno {i+1} - Informe Generado por IA:**\n{turn.get('synthesis', '')}")
        context.append("---")
    return "\n".join(context)

def get_initial_prompt(user_prompt, history: List[Dict[str, str]] = []):
    """Prompt general para Gemini y Claude, ahora con historial."""
    history_context = build_history_context(history)
    return f"""
{history_context}

**Instrucciones para la Respuesta Actual:**
1.  **Considera el Historial:** Basa tu respuesta en la nueva consulta del usuario, pero utiliza el historial anterior como contexto fundamental.
2.  **Idioma y Estilo:** Responde en español, de forma breve y directa. Ofrece los puntos clave o una respuesta inicial clara a la consulta actual.

**Consulta Actual del Usuario:**
"{user_prompt}"
"""

def get_deepseek_initial_prompt(user_prompt, history: List[Dict[str, str]] = []):
    """Prompt específico para DeepSeek, ahora con historial."""
    history_context = build_history_context(history)
    return f"""
{history_context}

**Rol:** Eres un analista de estrategias.
**Tarea para la Consulta Actual:**
Considerando el historial, tu objetivo es identificar y describir brevemente (2-3 puntos) las **nuevas perspectivas o ángulos** para analizar la consulta actual del usuario. No repitas análisis anteriores.

**Consulta Actual del Usuario:**
"{user_prompt}"
"""

# ... (El resto de omni_app.py, incluyendo las funciones de streaming y de síntesis, no necesita cambios) ...
# ... Las funciones build_summary_prompt y build_detailed_report_prompt ya usan el `user_prompt` que contendrá la pregunta de refinamiento ...

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
def build_summary_prompt(user_prompt, highlighted, normal):
    context = ""
    if highlighted:
        context += "**PERSPECTIVA DESTACADA (Fuente principal):**\n"
        for model, content in highlighted.items(): context += f"**Respuesta de {model.title()}:**\n{content}\n\n"
    if normal:
        context += "**PERSPECTIVAS ADICIONALES (Contexto secundario):**\n"
        for model, content in normal.items(): context += f"**Respuesta de {model.title()}:**\n{content}\n\n"
    
    return f"""
**Rol:** Eres un sintetizador experto.
**Tarea:** Basado en la consulta original del usuario y las perspectivas de IA proporcionadas, crea un **resumen final conciso y unificado**. Integra los puntos más fuertes de la perspectiva destacada y usa las adicionales para añadir matices. La respuesta debe ser fluida, clara y directa.
**Consulta Original:** "{user_prompt}"
**Perspectivas de IA:**
{context}
**COMIENZA TU RESUMEN FINAL AQUÍ:**
"""

def build_detailed_report_prompt(user_prompt, highlighted, normal):
    context = ""
    if highlighted:
        context += "**PERSPECTIVA DESTACADA (Fuente principal):**\n"
        for model, content in highlighted.items(): context += f"**Respuesta de {model.title()}:**\n{content}\n\n"
    if normal:
        context += "**PERSPECTIVAS ADICIONALES (Contexto secundario):**\n"
        for model, content in normal.items(): context += f"**Respuesta de {model.title()}:**\n{content}\n\n"
    
    return f"""
**Rol:** Actúa como un analista experto. Tu tarea es crear un **informe detallado**.
**Consulta Original:** "{user_prompt}"
**Análisis Preliminares:**
{context}
**Instrucciones para el Informe Detallado:**
Elabora un informe estructurado con las siguientes secciones obligatorias en Markdown:
1.  **Título Ejecutivo:** Claro y conciso.
2.  **Resumen Ejecutivo (TL;DR):** 2-3 frases con las conclusiones clave.
3.  **Análisis Comparativo:** Discute los puntos de convergencia y divergencia entre las IAs.
4.  **Síntesis Profunda:** Integra las perspectivas en una narrativa unificada, añadiendo conclusiones inferidas.
5.  **Puntos Clave / Accionables:** Una lista final de 3 a 5 puntos importantes.
"""

async def generate_synthesis_with_gemini(prompt):
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
    
    # MODIFICADO: Pasamos el historial a las funciones de prompt
    general_prompt = get_initial_prompt(request.prompt, request.history)
    deepseek_prompt = get_deepseek_initial_prompt(request.prompt, request.history)
    
    async def event_stream():
        tasks = {
            "gemini": stream_gemini(general_prompt),
            "deepseek": stream_deepseek(deepseek_prompt),
            "claude": stream_claude(general_prompt)
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

    if not highlighted and not normal:
         normal = {m: c['content'] for m, c in request.initial_responses.items() if request.decisions.get(m) != 'discard'}

    if request.synthesis_type == 'report':
        synthesis_prompt = build_detailed_report_prompt(request.prompt, highlighted, normal)
    else:
        synthesis_prompt = build_summary_prompt(request.prompt, highlighted, normal)
    
    try:
        final_synthesis = await generate_synthesis_with_gemini(synthesis_prompt)
        return {"synthesis": final_synthesis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "OmniQuery API v8.0 - FastAPI con Memoria Conversacional"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
