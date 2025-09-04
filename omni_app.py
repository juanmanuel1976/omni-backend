# ==============================================================================
# OMNIQUERY - SERVIDOR FASTAPI v9.1 (con Modos de Análisis Unificados)
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

# --- CONFIGURACIÓN E INICIALIZACIÓN (sin cambios) ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
app = FastAPI(title="OmniQuery API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- MODELOS PYDANTIC (sin cambios) ---
class GenerateRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []
    mode: Literal['direct', 'perspectives'] = 'direct'

class RefineRequest(BaseModel):
    prompt: str
    decisions: dict
    initial_responses: dict
    synthesis_type: Literal['summary', 'report'] = 'summary'

# --- SISTEMA DE PROMPTS (sin cambios) ---
def build_history_context(history: List[Dict[str, str]]) -> str:
    # ... (código de la función sin cambios) ...
def get_initial_prompt(user_prompt, history: List[Dict[str, str]] = []):
    # ... (código de la función sin cambios) ...
def get_deepseek_initial_prompt(user_prompt, history: List[Dict[str, str]] = []):
    # ... (código de la función sin cambios) ...
def build_summary_prompt(user_prompt, highlighted, normal):
    # ... (código de la función sin cambios) ...
def build_detailed_report_prompt(user_prompt, highlighted, normal):
    # ... (código de la función sin cambios) ...

# --- FUNCIONES DE STREAMING Y SÍNTESIS (sin cambios) ---
async def stream_gemini(prompt):
    # ... (código de la función sin cambios) ...
async def stream_deepseek(prompt):
    # ... (código de la función sin cambios) ...
async def stream_claude(prompt):
    # ... (código de la función sin cambios) ...
async def generate_synthesis_with_gemini(prompt):
    # ... (código de la función sin cambios) ...


# --- RUTAS DE LA APLICACIÓN ---
@app.post('/api/generate')
async def generate_initial_stream(request: GenerateRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    # ==================================================================
    # INICIO DE LA MODIFICACIÓN
    # ==================================================================
    
    # Lógica simplificada basada en el modo elegido por el usuario
    if request.mode == 'perspectives':
        # Modo Estratégico: TODAS las IAs reciben el prompt de estratega
        prompt_to_use = get_deepseek_initial_prompt(request.prompt, request.history)
    else: # mode == 'direct'
        # Modo Directo: TODAS las IAs reciben el prompt general para respuestas directas
        prompt_to_use = get_initial_prompt(request.prompt, request.history)

    async def event_stream():
        # Todas las IAs ahora usan el mismo prompt determinado por la lógica anterior
        tasks = {
            "gemini": stream_gemini(prompt_to_use),
            "deepseek": stream_deepseek(prompt_to_use),
            "claude": stream_claude(prompt_to_use)
        }
        
    # ==================================================================
    # FIN DE LA MODIFICACIÓN
    # ==================================================================
        
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


# --- El resto del archivo no cambia ---
@app.post('/api/refine')
async def refine_and_synthesize(request: RefineRequest):
    # ... (código de la función sin cambios) ...

@app.get("/")
async def root():
    return {"message": "OmniQuery API v9.1 - FastAPI con Modos de Análisis Unificados"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
