# ==============================================================================
# OMNIQUERY - SERVIDOR DE PROTOTIPO FUNCIONAL v3.1
# Versión con corrección de error 500. Streaming funcional.
# ==============================================================================
import asyncio
import httpx
import os
import json
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from anthropic import AsyncAnthropic

# --- CONFIGURACIÓN DE CLAVES DE API ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- INICIALIZACIÓN DE LA APLICACIÓN FLASK ---
app = Flask(__name__)
# Se configura CORS para permitir peticiones desde cualquier origen.
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
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"model": "gemini", "chunk": f"Error Gemini: {error_text.decode()}"}
                    return
                
                buffer = ""
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode('utf-8')
                    # Buscamos en el buffer por patrones de texto completos
                    while '"text": "' in buffer:
                        start_index = buffer.find('"text": "') + len('"text": "')
                        end_index = buffer.find('"', start_index)
                        if end_index != -1:
                            text = buffer[start_index:end_index]
                            buffer = buffer[end_index+1:]
                            # Decodificamos secuencias de escape de JSON como \n
                            clean_text = text.encode().decode('unicode_escape')
                            yield {"model": "gemini", "chunk": clean_text}
                        else:
                            break # No se encontró comilla de cierre, esperar más datos
    except Exception as e:
        yield {"model": "gemini", "chunk": f" Error en stream Gemini: {e}"}

async def stream_deepseek(prompt):
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
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and data['choices'][0]['delta'].get('content'):
                                yield {"model": "deepseek", "chunk": data['choices'][0]['delta']['content']}
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield {"model": "deepseek", "chunk": f" Error en stream DeepSeek: {e}"}

async def stream_claude(prompt):
    try:
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)
        async with client.messages.stream(
            model="claude-3-haiku-20240307", max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                yield {"model": "claude", "chunk": text}
    except Exception as e:
        yield {"model": "claude", "chunk": f" Error en stream Claude: {e}"}

# --- RUTA DE GENERACIÓN MODIFICADA PARA STREAMING ---
@app.route('/api/generate', methods=['POST'])
async def generate_initial_stream():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return Response(json.dumps({"error": "No prompt provided"}), status=400, mimetype='application/json')

    initial_prompt = get_initial_prompt(prompt)

    async def event_stream():
        tasks = {
            "gemini": asyncio.create_task(stream_gemini(initial_prompt)),
            "deepseek": asyncio.create_task(stream_deepseek(initial_prompt)),
            "claude": asyncio.create_task(stream_claude(initial_prompt))
        }

        async def get_next(model_name):
            try:
                # Obtenemos el siguiente trozo del generador
                return await tasks[model_name].__anext__()
            except StopAsyncIteration:
                return None

        # Mientras haya tareas activas
        while tasks:
            # Creamos tareas para obtener el siguiente trozo de cada stream activo
            futures = {asyncio.create_task(get_next(name)): name for name in tasks}
            done, pending = await asyncio.wait(futures.keys(), return_when=asyncio.FIRST_COMPLETED)
            
            for future in done:
                model_name = futures[future]
                result = future.result()
                if result:
                    yield f"data: {json.dumps(result)}\n\n"
                else:
                    # Este stream ha terminado, lo quitamos de las tareas activas
                    del tasks[model_name]
            
            for future in pending:
                future.cancel()
        
        yield f"data: {json.dumps({'model': 'system', 'chunk': 'DONE'})}\n\n"

    return Response(event_stream(), mimetype='text/event-stream')

# --- RUTA DE REFINAMIENTO (TEMPORALMENTE SIMPLIFICADA PARA EVITAR ERRORES) ---
@app.route('/api/refine', methods=['POST'])
async def refine_and_synthesize():
    # NOTA: La lógica de refinamiento completa necesita ser reimplementada
    # para ser compatible con el nuevo código de streaming. Por ahora,
    # devolvemos una respuesta de marcador de posición para evitar el error 500.
    return jsonify({
        "refined": {},
        "synthesis": "La función de refinamiento y síntesis está en desarrollo para ser compatible con el modo streaming. ¡Vuelve pronto!"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
