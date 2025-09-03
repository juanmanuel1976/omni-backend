# ==============================================================================
# OMNIQUERY - SERVIDOR DE PROTOTIPO FUNCIONAL v3.0
# Versión con Streaming (Server-Sent Events) para respuestas en tiempo real.
# ==============================================================================
import asyncio
import httpx
import os
import json
from flask import Flask, request, Response
from flask_cors import CORS
from anthropic import AsyncAnthropic

# --- CONFIGURACIÓN DE CLAVES DE API ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- INICIALIZACIÓN DE LA APLICACIÓN FLASK ---
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
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={GOOGLE_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"model": "gemini", "chunk": f"Error: {error_text.decode()}"}
                    return
                
                async for chunk in response.aiter_bytes():
                    chunk_str = chunk.decode('utf-8')
                    if chunk_str.strip().startswith('"text":'):
                        text = chunk_str.strip().replace('"text": "', '').replace('"', '').replace('\\n', '\n')
                        yield {"model": "gemini", "chunk": text}
    except Exception as e:
        yield {"model": "gemini", "chunk": f" Error en stream: {e}"}

async def stream_deepseek(prompt):
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "stream": True}
            async with client.stream("POST", "https://api.deepseek.com/chat/completions", headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"model": "deepseek", "chunk": f"Error: {error_text.decode()}"}
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
        yield {"model": "deepseek", "chunk": f" Error en stream: {e}"}

async def stream_claude(prompt):
    try:
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        async with client.messages.stream(
            model="claude-3-haiku-20240307", max_tokens=4096,
            messages=[{"role": "user", "content": prompt}], timeout=120.0
        ) as stream:
            async for text in stream.text_stream:
                yield {"model": "claude", "chunk": text}
    except Exception as e:
        yield {"model": "claude", "chunk": f" Error en stream: {e}"}

# --- RUTA DE GENERACIÓN MODIFICADA PARA STREAMING ---
@app.route('/api/generate', methods=['POST'])
async def generate_initial_stream():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return Response(json.dumps({"error": "No prompt provided"}), status=400, mimetype='application/json')

    initial_prompt = get_initial_prompt(prompt)

    async def event_stream():
        # Creamos una tarea para cada stream de IA
        tasks = [
            asyncio.create_task(stream_gemini(initial_prompt)),
            asyncio.create_task(stream_deepseek(initial_prompt)),
            asyncio.create_task(stream_claude(initial_prompt))
        ]

        # Usamos un truco para iterar sobre los generadores a medida que producen resultados
        async def drained_generator(gen):
            async for item in gen:
                yield item

        gens = [drained_generator(task) for task in tasks]

        # Mezclamos los resultados a medida que llegan
        while gens:
            # Esperamos a que alguno de los streams produzca un resultado
            done, pending = await asyncio.wait(
                [asyncio.create_task(gen.__anext__()) for gen in gens],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for future in done:
                try:
                    result = future.result()
                    # Enviamos el dato al frontend
                    yield f"data: {json.dumps(result)}\n\n"
                except StopAsyncIteration:
                    # Este generador ha terminado, lo quitamos de la lista
                    gens = [g for g in gens if g.__aiter__() is not future._coro.cr_frame.f_locals['self']]
            
            # Cancelamos las tareas que no terminaron (no debería pasar con FIRST_COMPLETED)
            for future in pending:
                future.cancel()
        
        # Señal de fin de stream
        yield f"data: {json.dumps({'model': 'system', 'chunk': 'DONE'})}\n\n"


    return Response(event_stream(), mimetype='text/event-stream')

# La ruta /api/refine se mantiene igual por ahora para simplificar.
# Implementar streaming en el refinamiento es posible pero añade más complejidad.
@app.route('/api/refine', methods=['POST'])
async def refine_and_synthesize():
    data = request.json
    prompt, decisions, initial_responses = data.get('prompt'), data.get('decisions'), data.get('initial_responses')
    
    spanish_instruction = "Asegúrate de que tu respuesta esté en español.\n\n"
    active_models = {k: v for k, v in decisions.items() if v != 'discard'}
    highlighted_model = next((k for k, v in decisions.items() if v == 'highlight'), None)
    
    tasks_r2 = []
    if highlighted_model:
        context = f"Respuesta destacada como modelo a seguir:\n{initial_responses[highlighted_model]['content']}"
        for model_name in active_models:
            if model_name == highlighted_model:
                tasks_r2.append(asyncio.sleep(0, result=(model_name.capitalize(), initial_responses[model_name]['content'])))
            else:
                new_prompt = f"{spanish_instruction}Pregunta Original: '{prompt}'.\nEnfoque preferido por el usuario:\n{context}\n\nAdapta tu respuesta a este enfoque, manteniendo el idioma español."
                tasks_r2.append(globals()[f'call_{model_name.capitalize()}'](new_prompt))
    else:
        for model_to_run in active_models:
            context_prompts = [f"Contexto de {name.capitalize()}:\n{resp['content']}" for name, resp in initial_responses.items() if name in active_models and name != model_to_run]
            context = "\n\n".join(context_prompts)
            new_prompt = f"{spanish_instruction}Pregunta Original: '{prompt}'.\n\nContexto de otras IAs:\n{context}\n\nRefina tu respuesta con esta información, manteniendo el idioma español."
            tasks_r2.append(globals()[f'call_{model_to_run.capitalize()}'](new_prompt))

    # Necesitamos las funciones call_* originales para refinar
    async def call_Gemini(prompt): return "Gemini", (await stream_gemini(prompt).__anext__())['chunk']
    async def call_DeepSeek(prompt): return "DeepSeek", (await stream_deepseek(prompt).__anext__())['chunk']
    async def call_Claude(prompt): return "Claude", (await stream_claude(prompt).__anext__())['chunk']

    refined_results = await asyncio.gather(*tasks_r2)
    refined_responses = {model.lower(): text for model, text in refined_results}
    
    synthesis_context = "\n\n".join([f"Respuesta de {name.capitalize()}:\n{text}" for name, text in refined_responses.items()])
    synthesis_prompt = f"Actúa como un analista experto y responde en español. Pregunta: '{prompt}'. Respuestas refinadas:\n{synthesis_context}\n\nCrea un informe final unificado, coherente y bien estructurado en español."
    
    _, synthesis_text = await call_Gemini(synthesis_prompt)
    
    return jsonify({"refined": refined_responses, "synthesis": synthesis_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
