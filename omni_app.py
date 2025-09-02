# ==============================================================================
# OMNIQUERY - SERVIDOR DE PROTOTIPO FUNCIONAL v2.0
# Versión optimizada para despliegue en Render.
# ==============================================================================
import asyncio
import httpx
import os # Para leer las claves de API de forma segura
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from anthropic import AsyncAnthropic

# --- CONFIGURACIÓN DE CLAVES DE API (DESDE EL ENTORNO) ---
# Render inyectará estas claves de forma segura.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- INICIALIZACIÓN DE LA APLICACIÓN FLASK ---
app = Flask(__name__)
CORS(app)

# --- FUNCIONES ASÍCRONAS PARA LLAMAR A LAS IAS ---
async def call_gemini(prompt):
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = await client.post(url, json=payload)
            if r.status_code != 200: return "Gemini", f"Error HTTP {r.status_code}: {r.text}"
            return "Gemini", r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e: return "Gemini", f"Error Detallado: {type(e).__name__} - {e}"

async def call_deepseek(prompt):
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post("https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]})
            if r.status_code != 200: return "DeepSeek", f"Error HTTP {r.status_code}: {r.text}"
            return "DeepSeek", r.json()["choices"][0]["message"]["content"]
    except Exception as e: return "DeepSeek", f"Error Detallado: {type(e).__name__} - {e}"

async def call_claude(prompt):
    try:
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        msg = await client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=4096,
            messages=[{"role": "user", "content": prompt}], timeout=90.0)
        return "Claude", msg.content[0].text
    except Exception as e: return "Claude", f"Error Detallado: {type(e).__name__} - {e}"

# --- RUTAS DE LA API DEL SERVIDOR ---
@app.route('/api/generate', methods=['POST'])
async def generate_initial():
    data = request.json
    prompt = data.get('prompt')
    if not prompt: return jsonify({"error": "No se proporcionó un prompt"}), 400
    tasks = [call_gemini(prompt), call_deepseek(prompt), call_claude(prompt)]
    results_list = await asyncio.gather(*tasks)
    responses = {model.lower(): {"content": text, "model": model} for model, text in results_list}
    return jsonify(responses)

@app.route('/api/refine', methods=['POST'])
async def refine_and_synthesize():
    data = request.json
    prompt, decisions, initial_responses = data.get('prompt'), data.get('decisions'), data.get('initial_responses')
    active_models = {k: v for k, v in decisions.items() if v != 'discard'}
    highlighted_model = next((k for k, v in decisions.items() if v == 'highlight'), None)
    tasks_r2 = []
    if highlighted_model:
        context = f"Respuesta destacada como modelo a seguir:\n{initial_responses[highlighted_model]['content']}"
        for model_name in active_models:
            if model_name == highlighted_model:
                tasks_r2.append(asyncio.sleep(0, result=(model_name.capitalize(), initial_responses[model_name]['content'])))
            else:
                new_prompt = f"Pregunta Original: '{prompt}'.\nEnfoque preferido por el usuario:\n{context}\n\nAdapta tu respuesta a este enfoque."
                tasks_r2.append(globals()[f'call_{model_name}'](new_prompt))
    else:
        for model_to_run in active_models:
            context_prompts = [f"Contexto de {name.capitalize()}:\n{resp['content']}" for name, resp in initial_responses.items() if name in active_models and name != model_to_run]
            context = "\n\n".join(context_prompts)
            new_prompt = f"Pregunta Original: '{prompt}'.\n\nContexto de otras IAs:\n{context}\n\nRefina tu respuesta con esta información."
            tasks_r2.append(globals()[f'call_{model_to_run}'](new_prompt))
    refined_results = await asyncio.gather(*tasks_r2)
    refined_responses = {model.lower(): text for model, text in refined_results}
    synthesis_context = "\n\n".join([f"Respuesta de {name.capitalize()}:\n{text}" for name, text in refined_responses.items()])
    synthesis_prompt = f"Actúa como un analista experto. Pregunta: '{prompt}'. Respuestas refinadas:\n{synthesis_context}\n\nCrea un informe final unificado y coherente."
    _, synthesis_text = await call_gemini(synthesis_prompt)
    return jsonify({"refined": refined_responses, "synthesis": synthesis_text})

if __name__ == '__main__':
    # Esta parte solo se usa para pruebas locales, Render usará el comando 'gunicorn'.
    app.run(host='0.0.0.0', port=10000)

