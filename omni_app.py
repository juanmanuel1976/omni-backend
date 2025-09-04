# ==============================================================================
# OMNIQUERY - SERVIDOR DE PROTOTIPO FUNCIONAL v3.0
# Versión con FastAPI, optimizada para despliegue y con Modo Debate.
# ==============================================================================
import asyncio
import httpx
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import AsyncAnthropic

# --- CONFIGURACIÓN DE CLAVES DE API (DESDE EL ENTORNO) ---
# Render inyectará estas claves de forma segura.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- INICIALIZACIÓN DE LA APLICACIÓN FASTAPI ---
app = FastAPI(
    title="OmniQuery API",
    description="El cerebro que potencia la plataforma OmniQuery para la síntesis de inteligencia colectiva."
)
CORS(app, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- MODELOS DE DATOS (PYDANTIC) ---
class GenerateRequest(BaseModel):
    prompt: str

class RefineRequest(BaseModel):
    prompt: str
    decisions: dict
    initial_responses: dict

class DebateRequest(BaseModel):
    prompt: str
    initial_responses: dict
    roles: dict = {}

# --- FUNCIONES DE LLAMADA A LAS IAS ---
async def call_ai_model(model_name: str, prompt: str):
    try:
        if model_name == "gemini":
            async with httpx.AsyncClient(timeout=120.0) as client:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                r = await client.post(url, json=payload)
                if r.status_code != 200: return f"Error HTTP {r.status_code}: {r.text}"
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        
        elif model_name == "deepseek":
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post("https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]})
                if r.status_code != 200: return f"Error HTTP {r.status_code}: {r.text}"
                return r.json()["choices"][0]["message"]["content"]

        elif model_name == "claude":
            client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            msg = await client.messages.create(
                model="claude-3-haiku-20240307", max_tokens=4096,
                messages=[{"role": "user", "content": prompt}], timeout=120.0)
            return msg.content[0].text

    except Exception as e:
        return f"Error Detallado al llamar a {model_name.capitalize()}: {type(e).__name__} - {e}"

# --- RUTAS DE LA API (ENDPOINTS) ---

@app.get("/")
def read_root():
    return {"status": "OmniQuery API está online."}

@app.post('/api/generate')
async def generate_initial(request: GenerateRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No se proporcionó un prompt")
    
    tasks = [
        call_ai_model("gemini", request.prompt),
        call_ai_model("deepseek", request.prompt),
        call_ai_model("claude", request.prompt)
    ]
    results = await asyncio.gather(*tasks)
    
    responses = {
        "gemini": {"content": results[0], "model": "Gemini"},
        "deepseek": {"content": results[1], "model": "DeepSeek"},
        "claude": {"content": results[2], "model": "Claude"}
    }
    return JSONResponse(content=responses)

@app.post('/api/refine')
async def refine_and_synthesize(request: RefineRequest):
    # Lógica de refinamiento dirigido por el usuario (Modo Estándar)
    active_models = {k: v for k, v in request.decisions.items() if v != 'discard'}
    highlighted_model = next((k for k, v in request.decisions.items() if v == 'highlight'), None)
    
    tasks_r2 = []
    if highlighted_model:
        context = f"Respuesta destacada como modelo a seguir:\n{request.initial_responses[highlighted_model]['content']}"
        for model_name in active_models:
            if model_name == highlighted_model:
                tasks_r2.append(asyncio.sleep(0, result=(model_name.capitalize(), request.initial_responses[model_name]['content'])))
            else:
                new_prompt = f"Pregunta Original: '{request.prompt}'.\nEnfoque preferido por el usuario:\n{context}\n\nAdapta tu respuesta a este enfoque."
                tasks_r2.append(call_ai_model(model_name, new_prompt))
    else: # Modo Ampliación
        for model_to_run in active_models:
            context_prompts = [f"Contexto de {name.capitalize()}:\n{resp['content']}" for name, resp in request.initial_responses.items() if name in active_models and name != model_to_run]
            context = "\n\n".join(context_prompts)
            new_prompt = f"Pregunta Original: '{request.prompt}'.\n\nContexto de otras IAs:\n{context}\n\nRefina tu respuesta con esta información."
            tasks_r2.append(call_ai_model(model_to_run, new_prompt))
            
    refined_results_tuples = await asyncio.gather(*tasks_r2)
    refined_responses = {model.lower(): text for model, text in refined_results_tuples}
    
    synthesis_context = "\n\n".join([f"Respuesta de {name.capitalize()}:\n{text}" for name, text in refined_responses.items()])
    synthesis_prompt = f"Actúa como un analista experto. Pregunta: '{request.prompt}'. Respuestas refinadas:\n{synthesis_context}\n\nCrea un informe final unificado y coherente."
    
    synthesis_text = await call_ai_model("gemini", synthesis_prompt)
    
    return JSONResponse(content={"refined": refined_responses, "synthesis": synthesis_text})


@app.post('/api/debate')
async def automated_debate(request: DebateRequest):
    # Lógica del Modo de Debate Avanzado
    responses = request.initial_responses

    # Fase 1: Crítica Cruzada
    critique_tasks = []
    for model_to_critique in responses.keys():
        context_prompts = [f"Respuesta de {name.capitalize()}:\n{resp['content']}" for name, resp in responses.items() if name != model_to_critique]
        context = "\n\n".join(context_prompts)
        critique_prompt = f"Tu rol es ser un analista crítico. La pregunta original fue: '{request.prompt}'.\n\nAnaliza las siguientes respuestas de otras IAs:\n{context}\n\nIdentifica al menos una debilidad, suposición o limitación clave en estas respuestas. Sé específico y constructivo en tu crítica."
        critique_tasks.append(call_ai_model(model_to_critique, critique_prompt))

    critiques = await asyncio.gather(*critique_tasks)
    critiques_map = dict(zip(responses.keys(), critiques))

    # Fase 2: Refutación y Revisión
    revision_tasks = []
    for model_to_revise in responses.keys():
        own_response = responses[model_to_revise]['content']
        received_critiques = "\n".join([f"- Crítica de {name.capitalize()}: {critique}" for name, critique in critiques_map.items() if name != model_to_revise])
        revision_prompt = f"La pregunta original fue: '{request.prompt}'.\nTu respuesta inicial fue:\n'{own_response}'\n\nHas recibido las siguientes críticas a tu respuesta:\n{received_critiques}\n\nConsiderando estas críticas, genera una versión revisada y mejorada de tu respuesta original. Puedes refutar las críticas o incorporar los puntos válidos."
        revision_tasks.append(call_ai_model(model_to_revise, revision_prompt))
    
    revised_responses_list = await asyncio.gather(*revision_tasks)
    revised_responses = dict(zip(responses.keys(), revised_responses_list))
    
    # Fase 3: Síntesis y Consenso
    synthesis_context = "\n\n".join([f"Respuesta revisada de {name.capitalize()}:\n{text}" for name, text in revised_responses.items()])
    synthesis_prompt = f"Actúa como un moderador experto. Se ha llevado a cabo un debate entre IAs sobre la pregunta: '{request.prompt}'.\n\nEstas son las respuestas finales revisadas después de una ronda de críticas:\n{synthesis_context}\n\nTu tarea es crear un informe final que sintetice los puntos de acuerdo, reconozca explícitamente los desacuerdos restantes y ofrezca una conclusión final consensuada y equilibrada."
    
    synthesis_text = await call_ai_model("gemini", synthesis_prompt)
    
    return JSONResponse(content={"revised": revised_responses, "synthesis": synthesis_text})

if __name__ == '__main__':
    import uvicorn
    # Esta parte es solo para pruebas locales. Render usará el comando 'gunicorn'.
    uvicorn.run(app, host='0.0.0.0', port=10000)

