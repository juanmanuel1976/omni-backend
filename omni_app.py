# ==============================================================================
# OMNIQUERY - SERVIDOR DE PROTOTIPO FUNCIONAL v3.4
# Versión con corrección final del orden de inicialización del wrapper.
# ==============================================================================
import asyncio
import httpx
import os
import json
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from anthropic import AsyncAnthropic
from asgiref.wsgi import WsgiToAsgi # Se importa el traductor

# --- CONFIGURACIÓN DE CLAVES DE API ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- INICIALIZACIÓN DE LA APLICACIÓN FLASK ---
# Primero, creamos la app de Flask normalmente
app = Flask(__name__)
CORS(app)

# --- PROMPT INICIAL MEJORADO ---
def get_initial_prompt(user_prompt):
    return f"""
**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Estilo Conciso:** Para esta primera respuesta, sé muy breve y directo. Ofrece un resumen ejecutivo, los puntos clave o una respuesta inicial clara. Evita introducciones largas y formalidades. El objetivo es dar una primera impresión rápida y útil.

**Consulta del Usuario:**
"{user_prompt}"
"""
# --- NUEVAS FUNCIONES DE STREAMING PARA CADA IA ---
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
        yield {"model": "gemini", "chunk": f" Error en stream Gemini: {e}"}

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
        yield {"model": "deepseek", "chunk": f" Error en stream DeepSeek: {e}"}

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
        yield {"model": "claude", "chunk": f" Error en stream Claude: {e}"}

# --- RUTAS DE LA APLICACIÓN ---
# Ahora definimos las rutas sobre el objeto 'app' de Flask
@app.route('/api/generate', methods=['POST'])
def generate_initial_stream():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return Response(json.dumps({"error": "No prompt provided"}), status=400, mimetype='application/json')
    initial_prompt = get_initial_prompt(prompt)
    
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
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/refine', methods=['POST'])
def refine_and_synthesize():
    return jsonify({
        "refined": {},
        "synthesis": "La función de refinamiento y síntesis está en desarrollo para ser compatible con el modo streaming. ¡Vuelve pronto!"
    })

# --- APLICACIÓN DEL WRAPPER DE COMPATIBILIDAD ---
# Después de definir TODAS las rutas, इenvolvemos" la app para el servidor.
app = WsgiToAsgi(app)
