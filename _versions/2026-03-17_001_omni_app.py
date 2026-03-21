# ==============================================================================
# OMNIQUERY - SERVIDOR FUNCIONAL v6.2.1 (VERSIÓN PRODUCCIÓN / MAIN)
# ==============================================================================
import asyncio
import httpx
import os
import json
import io
import pypdf
import logging
import sys
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
from anthropic import AsyncAnthropic
from rag_manager import rag_manager
from datetime import datetime
from ocr_processor import ocr_processor
import sqlite3
from contextlib import contextmanager

# CONFIGURACIÓN DE LOGGING DETALLADO
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

MODEL_TIMEOUT_VALIDATION = 55.0  

# --- CONFIGURACIÓN DE CLAVES DE API Y ENTORNO --
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# NUEVO: Identificador de entorno (Por defecto asume 'produccion' por seguridad)
ENVIRONMENT = os.environ.get("ENVIRONMENT", "produccion")

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
    isDocument: bool = False

class GenerateInitialRequest(BaseModel):
    prompt: str
    history: list = []
    model: str

class RefineRequest(BaseModel):
    prompt: str
    decisions: dict
    initial_responses: dict
    synthesis_type: str
    history: list = []

class DebateRequest(BaseModel):
    prompt: str
    history: list = []
    initial_responses: Optional[Dict[str, str]] = None
    dissidenceContext: Optional[Dict] = None
    isDocument: bool = False

class SemanticConsensusRequest(BaseModel):
    responses: Dict[str, str]

class FactCheckRequest(BaseModel):
    text_to_check: str
    original_query: str

# --- REGISTRADOR DE CONSULTAS EN SEGUNDO PLANO (CON IDENTIFICADOR) ---
async def log_user_query_supabase(endpoint: str, prompt: str, extra_info: dict = None):
    """Guarda silenciosamente las consultas. Incluye el entorno (Produccion/Dev)."""
    
    # Inyectamos el entorno en la información extra
    safe_extra_info = extra_info or {}
    safe_extra_info["entorno"] = ENVIRONMENT

    if not SUPABASE_URL or not SUPABASE_KEY:
        try:
            os.makedirs("logs", exist_ok=True)
            log_entry = {
                "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint": endpoint,
                "prompt": prompt,
                "extra": safe_extra_info
            }
            with open("logs/historial_consultas.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Fallo al guardar el log local: {e}")
        return

    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        payload = {
            "fecha_hora": datetime.now().isoformat(),
            "endpoint": endpoint,
            "prompt": prompt,
            "extra_info": safe_extra_info
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{SUPABASE_URL}/rest/v1/consultas_log", headers=headers, json=payload)
    except Exception as e:
        logger.error(f"Fallo al guardar el log en Supabase: {e}")

# --- FUNCIONES AUXILIARES RAG ---
async def get_text_from_files(files: List[UploadFile]) -> str:
    text = ""
    for file in files:
        try:
            logger.info(f"Procesando archivo: {file.filename}")
            file_content = await file.read()
            if file.content_type == 'application/pdf':
                try:
                    logger.info(f"Detectando si PDF está escaneado...")
                    is_scanned = ocr_processor.is_pdf_scanned(file_content)
                    if is_scanned:
                        ocr_text = await ocr_processor.extract_text_with_ocr(file_content, max_pages=100)
                        text += ocr_text + "\n\n"
                    else:
                        pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
                        for page in pdf_reader.pages:
                            page_text = page.extract_text() or ""
                            text += page_text
                except Exception as pdf_error:
                    logger.error(f"ERROR procesando PDF: {pdf_error}", exc_info=True)
                    raise
            elif file.content_type == 'text/plain':
                text += file_content.decode('utf-8')
        except Exception as file_error:
            logger.error(f"ERROR general con archivo: {file_error}", exc_info=True)
            raise
    return text

# --- LÓGICA DE PROMPTS ---
def build_contextual_prompt(user_prompt, history, mode, isDocument=False):
    history_context = ""
    if history:
        history_context += "**Historial de la Conversación Anterior (para dar contexto):**\n"
        for turn in history:
            history_context += f"- **Tu Consulta Anterior:** {turn.get('prompt', 'N/A')}\n"
            history_context += f"- **Nuestra Síntesis Anterior:** {turn.get('synthesis', 'N/A')}\n---\n"
    if isDocument:
        base_prompt = f"""**Instrucciones Clave:**
1.  **Fuente de Verdad Absoluta:** El usuario ha proporcionado un documento. Su contenido es la única fuente de verdad.
2.  **Tarea:** Basa tu respuesta exclusivamente en la información contenida en el documento. No añadas conocimiento externo ni verifiques los datos del documento.
3.  **Idioma:** Responde siempre y únicamente en español.
**Consulta del Usuario sobre el Documento:**
"{user_prompt}"
"""
        return f"{history_context}\n{base_prompt}" if history_context else base_prompt
    
    if mode == 'perspectives':
        base_prompt = f"""**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Análisis Estructurado:** Tu tarea principal es ser útil. Si la consulta pide datos concretos, primero establece la base factual de manera clara y precisa. Solo después, si es apropiado, desarrolla un análisis estratégico sobre esa base verificable.
**Consulta Actual del Usuario:**
"{user_prompt}"
"""
    else:
        base_prompt = f"""**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Estilo Conciso:** Sé muy breve y directo.
**Consulta Actual del Usuario:**
"{user_prompt}"
"""
    return f"{history_context}\n{base_prompt}" if history_context else base_prompt

def build_enhanced_dialectic_prompt(base_prompt, dissidence_context=None):
    enhanced_prompt = base_prompt
    if dissidence_context:
        user_refinement = dissidence_context.get('userRefinementPrompt', '')
        confidence_level = dissidence_context.get('confidenceLevel', 'balanced')
        target_consensus = dissidence_context.get('targetConsensus', 70)
        included_dissidences = dissidence_context.get('includedDissidences', [])
        excluded_dissidences = dissidence_context.get('excludedDissidences', [])

        refinement_section = f"\n\n**INSTRUCCIONES DE REFINAMIENTO DIRIGIDO POR EL USUARIO:**"
        refinement_section += f"\n- **Nivel de Confianza Objetivo:** Apunta a un análisis de nivel '{confidence_level.title()}', lo que implica un consenso conceptual superior al {target_consensus}%."
        if user_refinement:
            refinement_section += f"\n- **Directiva Principal del Usuario:** \"{user_refinement}\""
        if included_dissidences:
            dissidence_descriptions = [d.get('description', '') for d in included_dissidences]
            refinement_section += f"\n- **Disidencias a Resolver (Prioridad Alta):** Debes abordar y encontrar un punto medio en las siguientes áreas de desacuerdo: {'; '.join(dissidence_descriptions)}."
        if excluded_dissidences:
            excluded_descriptions = [d.get('description', '') for d in excluded_dissidences]
            refinement_section += f"\n- **Aspectos a Excluir o Evitar:** No profundices en los siguientes puntos: {'; '.join(excluded_descriptions)}."
            
        enhanced_prompt += refinement_section
    return enhanced_prompt

# --- FUNCIONES DE LLAMADA A LAS APIs ---
async def stream_gemini(prompt):
    if not GOOGLE_API_KEY:
        yield {"model": "gemini", "chunk": "Error: GOOGLE_API_KEY no configurada."}
        return
    try:
        async with httpx.AsyncClient(timeout=360.0) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?key={GOOGLE_API_KEY}"
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
        async with httpx.AsyncClient(timeout=360.0) as client:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
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
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=360.0)
        async with client.messages.stream(model="claude-3-haiku-20240307", max_tokens=4069, messages=[{"role": "user", "content": prompt}]) as stream:
            async for text in stream.text_stream:
                yield {"model": "claude", "chunk": text}
    except Exception as e:
        yield {"model": "claude", "chunk": f"Error: {e}"}

async def call_ai_model_no_stream(model_name: str, prompt: str, timeout: float = 360.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if model_name == "gemini":
                if not GOOGLE_API_KEY: return "Error: GOOGLE_API_KEY no configurada."
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                r = await client.post(url, json=payload)
                if r.status_code != 200: return f"Error HTTP {r.status_code}: {r.text}"
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif model_name == "deepseek":
                if not DEEPSEEK_API_KEY: return "Error: DEEPSEEK_API_KEY no configurada."
                headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
                payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
                r = await client.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload)
                if r.status_code != 200: return f"Error HTTP {r.status_code}: {r.text}"
                return r.json()["choices"][0]["message"]["content"]
            elif model_name == "claude":
                if not ANTHROPIC_API_KEY: return "Error: ANTHROPIC_API_KEY no configurada."
                client_anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=360.0)
                msg = await client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=4069, messages=[{"role": "user", "content": prompt}])
                return msg.content[0].text
    except Exception as e:
        return f"Error en {model_name}: {e}"

# --- FUNCIONES DE ANÁLISIS Y FACT-CHECKING ---
async def analyze_semantic_consensus_with_claude(responses):
    consensus_prompt = f"""Analiza el nivel de consenso conceptual entre estas tres respuestas sobre el mismo tema.

**Respuesta Gemini:**
{responses.get('gemini', '')}

**Respuesta DeepSeek:**
{responses.get('deepseek', '')}

**Respuesta Claude:**
{responses.get('claude', '')}

**Tu tarea como mediador neutral:**
1. Identifica conceptos centrales comunes (aunque usen palabras diferentes)
2. Detecta áreas de consenso real vs divergencia superficial
3. Calcula un porcentaje de consenso conceptual (0-100%)
4. Lista los puntos específicos de consenso fuerte

**Criterios de evaluación:**
- Consenso fuerte: Conceptos donde los 3 modelos coinciden sustancialmente
- Consenso moderado: Conceptos donde 2 modelos coinciden
- Divergencia: Conceptos donde hay desacuerdo real

**Formato de respuesta (JSON válido):**
{{
    "consensus_score": [número entre 0-100],
    "strong_consensus": ["concepto1", "concepto2"],
    "moderate_consensus": ["concepto3", "concepto4"],
    "divergence_areas": ["área1", "área2"],
    "explanation": "explicación breve de 1-2 líneas"
}}

Responde SOLO con el JSON válido, sin texto adicional."""
    try:
        consensus_analysis = await call_ai_model_no_stream('claude', consensus_prompt)
        json_start = consensus_analysis.find('{')
        json_end = consensus_analysis.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_content = consensus_analysis[json_start:json_end]
        else:
            json_content = consensus_analysis
        result = json.loads(json_content)
        if 'consensus_score' in result:
            result['consensus_score'] = max(0, min(100, int(result['consensus_score'])))
        return result
    except Exception as e:
        print(f"Error en análisis semántico: {e}")
        return { "consensus_score": 30, "strong_consensus": [], "moderate_consensus": [], "divergence_areas": ["Error de conexión"], "explanation": "Error en análisis semántico" }

async def perform_fact_check(text_to_check: str, original_query: str):
    try:
        extraction_prompt = f"""
Del siguiente texto, extrae una lista de hasta 5 afirmaciones factuales clave y verificables (nombres, fechas, cifras, cargos). No extraigas opiniones ni análisis.

Texto a analizar:
"{text_to_check}"

Responde únicamente con un objeto JSON válido con la clave "claims", que contenga una lista de strings.
Ejemplo de salida: {{"claims": ["Luis Caputo es el Ministro de Economía desde Diciembre 2023.", "Sergio Massa fue candidato a presidente."]}}
"""
        raw_claims = await call_ai_model_no_stream('claude', extraction_prompt)
        claims_data = json.loads(raw_claims[raw_claims.find('{'):raw_claims.rfind('}')+1])
        claims = claims_data.get("claims", [])
        if not claims:
            return {"status": "success", "report": "No se detectaron afirmaciones factuales específicas para verificar."}
        verification_tasks = []
        for claim in claims:
            verification_prompt = f"""
Evalúa la siguiente afirmación basada en tu conocimiento general y datos públicos. Responde con un JSON que contenga "status" ('Verificado', 'Incierto', 'Refutado') y una "correction" si es necesario.

Afirmación: "{claim}"
Contexto de la consulta original: "{original_query}"
"""
            verification_tasks.append(call_ai_model_no_stream('gemini', verification_prompt))
        results = await asyncio.gather(*verification_tasks)
        report_items = []
        for i, raw_result in enumerate(results):
            try:
                result_json = json.loads(raw_result[raw_result.find('{'):raw_result.rfind('}')+1])
                report_items.append({ "claim": claims[i], "status": result_json.get("status", "Error"), "correction": result_json.get("correction", "-") })
            except Exception:
                report_items.append({"claim": claims[i], "status": "Error de Verificación", "correction": "-"})
        return {"status": "success", "report": report_items}
    except Exception as e:
        return {"status": "error", "message": f"Fallo el proceso de fact-checking: {str(e)}"}


# --- RUTAS DE LA APLICACIÓN (ENDPOINTS) ---

@app.post("/api/rag-analysis")
async def rag_analysis_and_synthesize(raw_request: Request, background_tasks: BackgroundTasks, prompt: str = Form(...), history_json: str = Form("[]"), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se han subido archivos.")
    try:
        x_forwarded = raw_request.headers.get("X-Forwarded-For")
        client_ip = x_forwarded.split(",")[0].strip() if x_forwarded else (raw_request.client.host if raw_request.client else "127.0.0.1")

        background_tasks.add_task(log_user_query_supabase, "/api/rag-analysis", prompt, {"archivos": [f.filename for f in files], "ip_usuario": client_ip})

        history = json.loads(history_json) if history_json else []
        raw_text = await get_text_from_files(files)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Los documentos subidos no contienen texto extraíble.")
        
        text_size_kb = len(raw_text) / 1024
        DEEPSEEK_LIMIT_KB = 120 

        if text_size_kb < DEEPSEEK_LIMIT_KB:
            print(f"INFO: Documento ({text_size_kb:.2f} KB) por debajo del umbral. Usando análisis directo (Vía Rápida).")
            augmented_prompt = f"""**Consulta del Usuario:** "{prompt}"

**Contenido Completo de los Documento(s) para tu Análisis:**
---
{raw_text}
---
"""
            direct_request = DebateRequest(prompt=augmented_prompt, history=history, isDocument=True)
            result = await debate_and_synthesize(raw_request, direct_request, background_tasks)
            result["rag_metadata"] = {"strategy_used": "direct_analysis_bypass", "source_files": [f.filename for f in files]}
            return result
        else:
            print(f"INFO: Documento ({text_size_kb:.2f} KB) supera el umbral. Usando RAG completo (Vía Profunda).")
            await rag_manager.initialize()
            if not await rag_manager.index_documents(raw_text, {"file_names": [f.filename for f in files]}):
                raise HTTPException(status_code=500, detail="Error indexando documentos.")
            
            relevant_context = await rag_manager.get_context_for_query(prompt, max_context_length=4000)
            if not relevant_context:
                relevant_context = raw_text[:4000]

            augmented_prompt = f"""**Contexto Extraído (por relevancia semántica a tu consulta):**
{relevant_context}

**Consulta del Usuario:**
{prompt}"""
            rag_request = DebateRequest(prompt=augmented_prompt, history=history, isDocument=True)
            result = await debate_and_synthesize(raw_request, rag_request, background_tasks)
            
            rag_stats = rag_manager.get_stats()
            if isinstance(result, dict):
                result["rag_metadata"] = {"strategy_used": "full_rag_pipeline", "stats": rag_stats, "source_files": [f.filename for f in files]}
            
            rag_manager.clear_index()
            return result
    except Exception as e:
        rag_manager.clear_index()
        raise HTTPException(status_code=500, detail=f"Error en análisis RAG: {str(e)}")
        
@app.post('/api/generate')
async def generate_initial_stream(raw_request: Request, request: GenerateRequest, background_tasks: BackgroundTasks):
    x_forwarded = raw_request.headers.get("X-Forwarded-For")
    client_ip = x_forwarded.split(",")[0].strip() if x_forwarded else (raw_request.client.host if raw_request.client else "127.0.0.1")
    
    background_tasks.add_task(log_user_query_supabase, "/api/generate", request.prompt, {"mode": request.mode, "ip_usuario": client_ip})

    contextual_prompt = build_contextual_prompt(request.prompt, request.history, request.mode)
    async def event_stream():
        tasks = { "gemini": stream_gemini(contextual_prompt), "deepseek": stream_deepseek(contextual_prompt), "claude": stream_claude(contextual_prompt) }
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
            done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                agen = tasks.pop(task)
                try:
                    yield task.result()
                    tasks[asyncio.create_task(agen.__anext__())] = agen
                except StopAsyncIteration:
                    pass
    return StreamingResponse(event_stream(), media_type='text/event-stream')

@app.post('/api/generate-initial')
async def generate_initial_response(raw_request: Request, request: GenerateInitialRequest, background_tasks: BackgroundTasks):
    x_forwarded = raw_request.headers.get("X-Forwarded-For")
    client_ip = x_forwarded.split(",")[0].strip() if x_forwarded else (raw_request.client.host if raw_request.client else "127.0.0.1")
    
    background_tasks.add_task(log_user_query_supabase, "/api/generate-initial", request.prompt, {"model": request.model, "ip_usuario": client_ip})

    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct')
    response = await call_ai_model_no_stream(request.model, contextual_prompt)
    return {"response": response}

@app.post('/api/refine')
async def refine_and_synthesize(raw_request: Request, request: RefineRequest, background_tasks: BackgroundTasks):
    x_forwarded = raw_request.headers.get("X-Forwarded-For")
    client_ip = x_forwarded.split(",")[0].strip() if x_forwarded else (raw_request.client.host if raw_request.client else "127.0.0.1")
    
    background_tasks.add_task(log_user_query_supabase, "/api/refine", request.prompt, {"synthesis_type": request.synthesis_type, "ip_usuario": client_ip})

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
    synthesis_prompt_parts.append("\n**RESPUESTAS A CONSIDERAR:**")
    for model, content in active_responses.items():
        synthesis_prompt_parts.append(f"**Respuesta de {model.title()}:**\n{content}\n")
    final_prompt = "\n".join(synthesis_prompt_parts)
    synthesis_text = await call_ai_model_no_stream('gemini', final_prompt)
    return {"synthesis": synthesis_text}

@app.post('/api/debate')
async def debate_and_synthesize(raw_request: Request, request: DebateRequest, background_tasks: BackgroundTasks):
    x_forwarded = raw_request.headers.get("X-Forwarded-For")
    client_ip = x_forwarded.split(",")[0].strip() if x_forwarded else (raw_request.client.host if raw_request.client else "127.0.0.1")
    
    background_tasks.add_task(log_user_query_supabase, "/api/debate", request.prompt, {"isDocument": request.isDocument, "ip_usuario": client_ip})

    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct', request.isDocument)
    
    if request.dissidenceContext:
        contextual_prompt = build_enhanced_dialectic_prompt(contextual_prompt, request.dissidenceContext)

    if request.initial_responses:
        initial_responses = {k: v.get('content', '') if isinstance(v, dict) else v for k, v in request.initial_responses.items()}
    else:
        initial_tasks = [call_ai_model_no_stream(m, contextual_prompt) for m in ['gemini', 'deepseek', 'claude']]
        results = await asyncio.gather(*initial_tasks)
        initial_responses = {'gemini': results[0], 'deepseek': results[1], 'claude': results[2]}

    critique_prompts = {}
    models_order = ['gemini', 'deepseek', 'claude']
    is_refinement_iteration = bool(request.dissidenceContext and request.dissidenceContext.get('userRefinementPrompt'))

    for model in models_order:
        context = "\n\n".join([f"**Respuesta de {m.title()}:**\n{r}" for m, r in initial_responses.items() if m != model])
        if not is_refinement_iteration:
            base_critique = f"""**Tu Respuesta Inicial:**\n{initial_responses[model]}\n\n**Respuestas de Colegas:**\n{context}\n\n**Tu Tarea (Ronda 1 - Crítica Abierta):** Analiza críticamente las respuestas de tus colegas. Identifica fortalezas, debilidades y puntos ciegos. Refina y mejora tu propio argumento incorporando las perspectivas valiosas para enriquecer el análisis global."""
        else:
            base_critique = f"""**Tu Respuesta Anterior:**\n{initial_responses[model]}\n\n**Respuestas de Colegas (Ronda Anterior):**\n{context}\n\n**Tu Tarea (Ronda de Refinamiento):** El usuario ha dado nuevas instrucciones (detalladas en la consulta principal). Tu objetivo es integrar estas directivas. Reformula tu análisis para alinearte con la guía del usuario, manteniendo los consensos ya logrados y abordando las diferencias críticas señaladas."""
        critique_prompts[model] = base_critique
    
    critique_tasks = [call_ai_model_no_stream(m, critique_prompts[m]) for m in models_order]
    revised_results = await asyncio.gather(*critique_tasks)
    revised_responses = dict(zip(models_order, revised_results))
    synthesis_context = "\n\n".join([f"**Argumento Revisado de {m.title()}:**\n{r}" for m, r in revised_responses.items()])
    synthesis_prompt = f"**Consulta Original (con historial y directivas de refinamiento):**\n{contextual_prompt}\n\n**Debate de Expertos (Argumentos Revisados):**\n{synthesis_context}\n\n**Tu Tarea Final como Moderador:** Eres un experto en síntesis estratégica. Tu objetivo es crear un informe final unificado y coherente. Integra los argumentos revisados de los expertos en una única respuesta. Asegúrate de seguir TODAS las instrucciones y directivas de refinamiento dadas en la consulta original. La síntesis debe ser clara, accionable y responder directamente a la petición del usuario."

    if request.dissidenceContext and request.dissidenceContext.get('forcedSynthesis'):
        synthesis_prompt += "\n\n**INSTRUCCIÓN ESPECIAL DE SÍNTESIS FORZADA:** El usuario ha solicitado finalizar el debate. Enfócate en los consensos existentes y presenta las diferencias restantes como perspectivas complementarias o áreas para futura exploración, no como conflictos a resolver. El objetivo es entregar un resultado accionable con la información disponible."

    final_synthesis = await call_ai_model_no_stream('gemini', synthesis_prompt)
    return { "revised": revised_responses, "synthesis": final_synthesis, "initial": initial_responses, "dissidenceContext": request.dissidenceContext }

@app.post('/api/semantic-consensus')
async def semantic_consensus_endpoint(request: SemanticConsensusRequest):
    try:
        result = await analyze_semantic_consensus_with_claude(request.responses)
        return result
    except Exception as e:
        return { "consensus_score": 25, "strong_consensus": [], "moderate_consensus": [], "divergence_areas": ["Error del servidor"], "explanation": f"Error: {str(e)}" }

@app.post("/api/fact-check")
async def fact_check_endpoint(request: FactCheckRequest):
    return await perform_fact_check(request.text_to_check, request.original_query)

# --- RUTAS DE SALUD Y RAÍZ ---
@app.get("/")
async def root():
    return {"message": "Crisalia API v6.2.1 - Verificado y Completo"}

@app.get("/health")
async def health_check():
    """Endpoint de salud para monitoreo"""
    return {
        "status": "healthy",
        "version": "6.2.1",
        "features": ["dialectic_enhancements", "user_directed_refinement", "rag_pipeline"]
    }
@app.get("/api/rag-stats")
async def get_rag_stats():
    """Endpoint para monitorear el estado del RAG Manager."""
    return {
        "status": "healthy", 
        "rag_stats": rag_manager.get_stats(),
        "memory_usage": "optimized_for_standard_plan"
    }

@app.on_event("startup")
async def startup_event():
    """Inicialización optimizada del RAG Manager al startup."""
    try:
        print("Inicializando RAG Manager en startup...")
        await rag_manager.initialize()
        print("RAG Manager listo para análisis de documentos")
    except Exception as e:
        print(f"Error inicializando RAG Manager: {e}")
        
# ==============================================================================
# NUEVOS ENDPOINTS - Agente de Mejora Continua v1.0
# ==============================================================================

# --- MODELOS DE DATOS ---

class ValidateChangeRequest(BaseModel):
    archivo: str                    
    descripcion: str                
    codigo_original: str            
    codigo_propuesto: str           
    tipo: str = "code"              
    contexto_adicional: str = ""    

class ValidateChangeResponse(BaseModel):
    aprobado: bool
    consenso_score: int             
    veredicto: str                  
    debate: Dict[str, Any]          
    sintesis: str                   
    riesgos: list                   
    sugerencias: list               
    timestamp: str

class AgentStepRequest(BaseModel):
    paso_numero: int
    titulo: str
    descripcion: str
    archivos_afectados: list = []
    resultado: str = ""             
    validacion_id: str = ""         

class AgentStepResponse(BaseModel):
    paso_id: str
    timestamp: str
    documentacion_actualizada: bool

# --- PERSISTENCIA SQLite ---
AGENT_DB_PATH = 'crisalia_agent.db'

@contextmanager
def get_agent_db():
    conn = sqlite3.connect(AGENT_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_agent_db():
    with get_agent_db() as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS agent_steps '
            '(paso_id TEXT PRIMARY KEY, paso_numero INTEGER, titulo TEXT, '
            'descripcion TEXT, archivos_afectados TEXT, resultado TEXT, '
            'validacion_id TEXT, timestamp TEXT)'
        )
        conn.execute(
            'CREATE TABLE IF NOT EXISTS validation_logs '
            '(id TEXT PRIMARY KEY, timestamp TEXT, archivo TEXT, tipo TEXT, '
            'descripcion TEXT, veredicto TEXT, consenso_score INTEGER, debate TEXT)'
        )

init_agent_db()

_agent_log: list = []
_validation_log: list = []

# --- ENDPOINT 1: VALIDACIÓN DIALÉCTICA ---

@app.post("/api/validate-change", response_model=ValidateChangeResponse)
async def validate_change(request: ValidateChangeRequest):
    tipo_label = {
        "code": "corrección/optimización de código",
        "ux": "mejora de interfaz de usuario",
        "benchmark": "cambio en métricas de evaluación"
    }.get(request.tipo, "cambio")

    prompt_validacion = f"""Sos un experto en desarrollo de software analizando una propuesta de {tipo_label} para Crisalia.

Crisalia es una plataforma multi-IA dialéctica para profesionales que necesitan análisis robustos y libres del sesgo de un modelo único.
Su diferencial es hacer debatir a Gemini, DeepSeek y Claude para producir síntesis superiores.
Sus casos de uso incluyen análisis estratégico, investigación, due diligence legal y financiero, y toma de decisiones de alto impacto en cualquier industria.

ARCHIVO AFECTADO: {request.archivo}

DESCRIPCIÓN DEL CAMBIO:
{request.descripcion}

CÓDIGO ORIGINAL:
---
{request.codigo_original}
---

CÓDIGO PROPUESTO:
---
{request.codigo_propuesto}
---

{f"CONTEXTO ADICIONAL: {request.contexto_adicional}" if request.contexto_adicional else ""}

TU TAREA:
1. Analizá si el cambio propuesto es correcto y seguro
2. Identificá riesgos concretos (efectos secundarios, casos edge, dependencias rotas)
3. Sugerí mejoras si las hay
4. Evaluá el impacto en la monetización/usabilidad para clientes jurídicos

Respondé de forma concisa y técnica. Máximo 4 oraciones."""

    initial_tasks = [
        call_ai_model_no_stream('gemini', prompt_validacion, timeout=MODEL_TIMEOUT_VALIDATION),
        call_ai_model_no_stream('deepseek', prompt_validacion, timeout=MODEL_TIMEOUT_VALIDATION),
        call_ai_model_no_stream('claude', prompt_validacion, timeout=MODEL_TIMEOUT_VALIDATION)
    ]
    results = await asyncio.gather(*initial_tasks)
    initial_responses = {
        'gemini': results[0],
        'deepseek': results[1],
        'claude': results[2]
    }

    critique_tasks = []
    models_order = ['gemini', 'deepseek', 'claude']
    for model in models_order:
        context = "\n\n".join([
            f"**{m.title()} dijo:**\n{r}"
            for m, r in initial_responses.items() if m != model
        ])
        critique_prompt = f"""{prompt_validacion}

**Tu respuesta inicial fue:**
{initial_responses[model]}

**Los otros modelos dijeron:**
{context}

**Ahora:** ¿Coincidís con ellos? ¿Qué agregarías o corregirías? ¿Hay riesgos que no mencionaron?
Sé específico. Si el cambio es seguro, decilo claramente. Si hay riesgo, nombrá el riesgo exacto."""
        critique_tasks.append(call_ai_model_no_stream(model, critique_prompt, timeout=MODEL_TIMEOUT_VALIDATION))

    critique_results = await asyncio.gather(*critique_tasks)
    revised_responses = dict(zip(models_order, critique_results))

    synthesis_prompt = f"""Sos el árbitro final de una revisión de código dialéctica para Crisalia.

CAMBIO PROPUESTO en {request.archivo}:
{request.descripcion}

DEBATE DE LOS 3 MODELOS:
Gemini: {revised_responses['gemini']}
DeepSeek: {revised_responses['deepseek']}
Claude: {revised_responses['claude']}

Producí un veredicto final en formato JSON válido:
{{
    "aprobado": true/false,
    "consenso_score": 0-100,
    "veredicto": "APROBADO" | "APROBADO_CON_OBSERVACIONES" | "RECHAZADO",
    "sintesis": "2-3 oraciones explicando el veredicto",
    "riesgos": ["riesgo1", "riesgo2"],
    "sugerencias": ["sugerencia1", "sugerencia2"]
}}

Respondé SOLO con el JSON válido, sin texto adicional."""

    raw_verdict = await call_ai_model_no_stream('gemini', synthesis_prompt, timeout=MODEL_TIMEOUT_VALIDATION)

    try:
        json_start = raw_verdict.find('{')
        json_end = raw_verdict.rfind('}') + 1
        verdict = json.loads(raw_verdict[json_start:json_end])
    except Exception:
        verdict = {
            "aprobado": False,
            "consenso_score": 0,
            "veredicto": "ERROR_PARSEO",
            "sintesis": raw_verdict[:500],
            "riesgos": ["Error al parsear el veredicto"],
            "sugerencias": []
        }

    validation_entry = {
        "id": f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "archivo": request.archivo,
        "tipo": request.tipo,
        "descripcion": request.descripcion,
        "veredicto": verdict.get("veredicto"),
        "consenso_score": verdict.get("consenso_score", 0),
        "debate": {
            "inicial": initial_responses,
            "revisado": revised_responses
        }
    }
    _validation_log.append(validation_entry)

    return ValidateChangeResponse(
        aprobado=verdict.get("aprobado", False),
        consenso_score=verdict.get("consenso_score", 0),
        veredicto=verdict.get("veredicto", "ERROR"),
        debate={"inicial": initial_responses, "revisado": revised_responses},
        sintesis=verdict.get("sintesis", ""),
        riesgos=verdict.get("riesgos", []),
        sugerencias=verdict.get("sugerencias", []),
        timestamp=validation_entry["timestamp"]
    )


# --- ENDPOINT 2: REGISTRO DE PASOS DEL AGENTE ---

@app.post("/api/agent-step", response_model=AgentStepResponse)
async def register_agent_step(request: AgentStepRequest):
    paso_id = f"step_{request.paso_numero:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.now().isoformat()

    step_entry = {
        "paso_id": paso_id,
        "paso_numero": request.paso_numero,
        "titulo": request.titulo,
        "descripcion": request.descripcion,
        "archivos_afectados": request.archivos_afectados,
        "resultado": request.resultado,
        "validacion_id": request.validacion_id,
        "timestamp": timestamp
    }
    _agent_log.append(step_entry)

    return AgentStepResponse(
        paso_id=paso_id,
        timestamp=timestamp,
        documentacion_actualizada=True
    )


# --- ENDPOINT 3: DOCUMENTACIÓN ACUMULADA ---

@app.get("/api/agent-docs")
async def get_agent_docs():
    total_validaciones = len(_validation_log)
    aprobadas = sum(1 for v in _validation_log if v.get("veredicto") == "APROBADO")
    con_obs = sum(1 for v in _validation_log if v.get("veredicto") == "APROBADO_CON_OBSERVACIONES")
    rechazadas = sum(1 for v in _validation_log if v.get("veredicto") == "RECHAZADO")

    return {
        "version": "1.0",
        "generado": datetime.now().isoformat(),
        "estadisticas": {
            "total_pasos": len(_agent_log),
            "total_validaciones": total_validaciones,
            "aprobadas": aprobadas,
            "aprobadas_con_observaciones": con_obs,
            "rechazadas": rechazadas,
            "tasa_aprobacion": round((aprobadas + con_obs) / total_validaciones * 100, 1) if total_validaciones > 0 else 0
        },
        "pasos": _agent_log,
        "validaciones": [
            {
                "id": v["id"],
                "timestamp": v["timestamp"],
                "archivo": v["archivo"],
                "descripcion": v["descripcion"],
                "veredicto": v["veredicto"],
                "consenso_score": v["consenso_score"]
            }
            for v in _validation_log
        ]
    }


# --- ENDPOINT 4: HEALTH EXTENDIDO ---

@app.get("/api/agent-status")
async def get_agent_status():
    return {
        "agente_activo": True,
        "version_agente": "1.0",
        "pasos_registrados": len(_agent_log),
        "validaciones_realizadas": len(_validation_log),
        "ultimo_paso": _agent_log[-1]["titulo"] if _agent_log else "ninguno",
        "ultima_validacion": _validation_log[-1]["veredicto"] if _validation_log else "ninguna"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
