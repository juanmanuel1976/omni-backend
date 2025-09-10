# ==============================================================================
# OMNIQUERY - SERVIDOR FUNCIONAL v6.0 (Con Mejoras Dialécticas Corregidas)
# ==============================================================================
import asyncio
import httpx
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
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

class GenerateInitialRequest(BaseModel):
    prompt: str
    history: list = []
    model: str  # 'gemini', 'deepseek', o 'claude'

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
    dissidenceContext: Optional[Dict] = None  # NUEVA FUNCIONALIDAD

# --- LÓGICA DE PROMPTS ---
def build_contextual_prompt(user_prompt, history, mode):
    history_context = ""
    if history:
        history_context += "**Historial de la Conversación Anterior (para dar contexto):**\n"
        for turn in history:
            history_context += f"- **Tu Consulta Anterior:** {turn.get('prompt', 'N/A')}\n"
            history_context += f"- **Nuestra Síntesis Anterior:** {turn.get('synthesis', 'N/A')}\n---\n"
    
    base_prompt = ""
    if mode == 'perspectives':
        base_prompt = f"""
**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Enfoque Estratégico:** No des una respuesta directa. Tu objetivo es explorar el tema desde un ángulo único o estratégico.
**Consulta Actual del Usuario:**
"{user_prompt}"
"""
    else:  # direct
        base_prompt = f"""
**Instrucciones Clave:**
1.  **Idioma Obligatorio:** Responde siempre y únicamente en español.
2.  **Estilo Conciso:** Sé muy breve y directo.
**Consulta Actual del Usuario:**
"{user_prompt}"
"""
    
    return f"{history_context}\n{base_prompt}" if history_context else base_prompt

def build_enhanced_dialectic_prompt(base_prompt, dissidence_context=None):
    """Construye un prompt mejorado para el modo dialéctico con contexto de disidencias"""
    enhanced_prompt = base_prompt
    
    if dissidence_context:
        user_refinement = dissidence_context.get('userRefinementPrompt', '')
        confidence_level = dissidence_context.get('confidenceLevel', 'balanced')
        target_consensus = dissidence_context.get('targetConsensus', 70)
        included_dissidences = dissidence_context.get('includedDissidences', [])
        excluded_dissidences = dissidence_context.get('excludedDissidences', [])
        
        refinement_section = f"""

**INSTRUCCIONES DE REFINAMIENTO:**
- **Nivel de Confianza Objetivo:** {confidence_level.title()} (>{target_consensus}% consenso)
- **Orientación del Usuario:** {user_refinement}
"""
        
        if included_dissidences:
            dissidence_descriptions = [d.get('description', '') for d in included_dissidences]
            refinement_section += f"""
- **Disidencias a Abordar:** {'; '.join(dissidence_descriptions)}
"""
        
        if excluded_dissidences:
            excluded_descriptions = [d.get('description', '') for d in excluded_dissidences]
            refinement_section += f"""
- **Aspectos a Evitar:** {'; '.join(excluded_descriptions)}
"""
        
        enhanced_prompt += refinement_section
    
    return enhanced_prompt

# --- FUNCIONES DE LLAMADA A LAS APIs ---
async def stream_gemini(prompt):
    if not GOOGLE_API_KEY:
        yield {"model": "gemini", "chunk": "Error: GOOGLE_API_KEY no configurada."}
        return
    try:
        async with httpx.AsyncClient(timeout=360.0) as client:  # TIMEOUT EXTENDIDO A 6 MINUTOS
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={GOOGLE_API_KEY}"
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
        async with httpx.AsyncClient(timeout=360.0) as client:  # TIMEOUT EXTENDIDO A 6 MINUTOS
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
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=360.0)  # TIMEOUT EXTENDIDO A 6 MINUTOS
        async with client.messages.stream(model="claude-3-haiku-20240307", max_tokens=4096, messages=[{"role": "user", "content": prompt}]) as stream:
            async for text in stream.text_stream:
                yield {"model": "claude", "chunk": text}
    except Exception as e:
        yield {"model": "claude", "chunk": f"Error: {e}"}

async def call_ai_model_no_stream(model_name: str, prompt: str):
    try:
        async with httpx.AsyncClient(timeout=360.0) as client:  # TIMEOUT EXTENDIDO A 6 MINUTOS
            if model_name == "gemini":
                if not GOOGLE_API_KEY: return "Error: GOOGLE_API_KEY no configurada."
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
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
                client_anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=360.0)  # TIMEOUT EXTENDIDO A 6 MINUTOS
                msg = await client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=4096, messages=[{"role": "user", "content": prompt}])
                return msg.content[0].text
    except Exception as e:
        return f"Error en {model_name}: {e}"

# ==============================================================================
# MODIFICACIONES PARA omni_app.py - Análisis Semántico con Claude
# ==============================================================================

# 1. AGREGAR ESTA FUNCIÓN después de la función call_ai_model_no_stream (línea ~140)

async def analyze_semantic_consensus_with_claude(responses):
    """Usa Claude para analizar consenso semántico entre respuestas"""
    
    # Construir prompt para análisis de consenso
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
        # Usar Claude para análisis semántico
        consensus_analysis = await call_ai_model_no_stream('claude', consensus_prompt)
        
        # Limpiar respuesta por si tiene texto extra
        json_start = consensus_analysis.find('{')
        json_end = consensus_analysis.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_content = consensus_analysis[json_start:json_end]
        else:
            json_content = consensus_analysis
        
        # Parsear respuesta JSON
        result = json.loads(json_content)
        
        # Validar que el score esté en rango válido
        if 'consensus_score' in result:
            result['consensus_score'] = max(0, min(100, result['consensus_score']))
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {consensus_analysis}")
        # Fallback con score básico
        return {
            "consensus_score": 50,
            "strong_consensus": [],
            "moderate_consensus": [],
            "divergence_areas": ["Error en análisis"],
            "explanation": "Error al procesar análisis semántico"
        }
    except Exception as e:
        print(f"Error en análisis semántico: {e}")
        # Fallback básico
        return {
            "consensus_score": 30,
            "strong_consensus": [],
            "moderate_consensus": [],
            "divergence_areas": ["Error de conexión"],
            "explanation": "Error en análisis semántico"
        }

# 2. AGREGAR NUEVO MODELO para el request (después de DebateRequest, línea ~55)

class SemanticConsensusRequest(BaseModel):
    responses: Dict[str, str]

# 3. AGREGAR NUEVO ENDPOINT (al final de los endpoints, antes de @app.get("/"), línea ~280)

@app.post('/api/semantic-consensus')
async def semantic_consensus_endpoint(request: SemanticConsensusRequest):
    """Endpoint para análisis semántico de consenso usando Claude como mediador"""
    try:
        result = await analyze_semantic_consensus_with_claude(request.responses)
        return result
    except Exception as e:
        return {
            "consensus_score": 25,
            "strong_consensus": [],
            "moderate_consensus": [],
            "divergence_areas": ["Error del servidor"],
            "explanation": f"Error: {str(e)}"
        }

# ==============================================================================
# MODIFICACIONES PARA dialectic-enhancements.js
# ==============================================================================

# REEMPLAZAR la función calculateAndDisplayConsensusEnhanced completa por esta versión:

async function calculateAndDisplayConsensusEnhanced() {
    try {
        // Preparar respuestas para análisis semántico
        const responsesForAnalysis = {};
        Object.keys(state.initial_responses).forEach(model => {
            responsesForAnalysis[model] = state.initial_responses[model].content || '';
        });
        
        // Verificar que tenemos respuestas válidas
        const hasValidResponses = Object.values(responsesForAnalysis).some(content => 
            content && content.trim().length > 50
        );
        
        if (!hasValidResponses) {
            // Fallback si no hay respuestas suficientes
            return calculateTraditionalConsensus();
        }
        
        // Llamar al backend para análisis semántico con Claude
        const response = await fetch(`${window.API_BASE_URL}/api/semantic-consensus`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                responses: responsesForAnalysis
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const semanticResult = await response.json();
        const score = semanticResult.consensus_score || 0;
        
        // Actualizar UI con análisis semántico detallado
        updateSemanticConsensusDisplay(score, semanticResult);
        
        // Ejecutar lógica de bajo consenso si es necesario
        const selectedMode = document.querySelector('input[name="analysisMode"]:checked')?.value;
        if (selectedMode === 'debate' && window.dialecticEnhancements) {
            window.dialecticEnhancements.handleAdvancedLowConsensus(state.initial_responses, score);
        } else if (score < 40) {
            handleLowConsensus();
        }
        
        return score;
        
    } catch (error) {
        console.error('Error en análisis semántico:', error);
        // Fallback al método tradicional
        return calculateTraditionalConsensus();
    }
}

function updateSemanticConsensusDisplay(score, analysis) {
    const consensusContainer = document.getElementById('consensus-container');
    const consensusScore = document.getElementById('consensus-score');
    const consensusBar = document.getElementById('consensus-bar');
    const consensusStatus = document.getElementById('consensus-status');
    
    if (consensusContainer) consensusContainer.classList.remove('hidden');
    if (consensusScore) consensusScore.textContent = `${score}%`;
    if (consensusBar) {
        setTimeout(() => { consensusBar.style.width = `${score}%`; }, 100);
    }
    
    // Mostrar análisis detallado
    if (consensusStatus && analysis) {
        const strongCount = analysis.strong_consensus ? analysis.strong_consensus.length : 0;
        const moderateCount = analysis.moderate_consensus ? analysis.moderate_consensus.length : 0;
        
        consensusStatus.innerHTML = `
            <div class="text-sm text-gray-300 mt-2">
                <div class="mb-1">Análisis semántico: ${strongCount} consensos fuertes, ${moderateCount} moderados</div>
                ${analysis.explanation ? `<div class="text-xs text-gray-400 mb-2">${analysis.explanation}</div>` : ''}
                <details class="cursor-pointer">
                    <summary class="text-violet-400 hover:text-violet-300">Ver desglose detallado</summary>
                    <div class="mt-2 text-xs space-y-1 bg-gray-900/50 p-2 rounded">
                        ${strongCount > 0 ? `<div><strong class="text-green-400">Consenso fuerte:</strong> ${analysis.strong_consensus.join(', ')}</div>` : ''}
                        ${moderateCount > 0 ? `<div><strong class="text-yellow-400">Consenso moderado:</strong> ${analysis.moderate_consensus.join(', ')}</div>` : ''}
                        ${analysis.divergence_areas && analysis.divergence_areas.length > 0 ? `<div><strong class="text-red-400">Divergencias:</strong> ${analysis.divergence_areas.join(', ')}</div>` : ''}
                    </div>
                </details>
            </div>
        `;
    }
}

function calculateTraditionalConsensus() {
    // Función de fallback (mantener la existente)
    const keywordsGemini = getKeywords(state.initial_responses.gemini.content);
    const keywordsDeepseek = getKeywords(state.initial_responses.deepseek.content);
    const keywordsClaude = getKeywords(state.initial_responses.claude.content);
    
    const allSets = [keywordsGemini, keywordsDeepseek, keywordsClaude].filter(s => s.size > 0);
    if (allSets.length < 2) return 0;
    
    const intersection = new Set([...allSets[0]].filter(word => allSets.every(set => set.has(word))));
    const union = new Set([...keywordsGemini, ...keywordsDeepseek, ...keywordsClaude]);
    
    return union.size === 0 ? 0 : Math.round((intersection.size / union.size) * 100);
}
# --- RUTAS DE LA APLICACIÓN ---
@app.post('/api/generate')
async def generate_initial_stream(request: GenerateRequest):
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, request.mode)
    
    async def event_stream():
        tasks = {
            "gemini": stream_gemini(contextual_prompt),
            "deepseek": stream_deepseek(contextual_prompt),
            "claude": stream_claude(contextual_prompt)
        }
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
async def generate_initial_response(request: GenerateInitialRequest):
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct')
    response = await call_ai_model_no_stream(request.model, contextual_prompt)
    return {"response": response}

@app.post('/api/refine')
async def refine_and_synthesize(request: RefineRequest):
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
async def debate_and_synthesize(request: DebateRequest):
    contextual_prompt = build_contextual_prompt(request.prompt, request.history, 'direct')
    
    # NUEVA FUNCIONALIDAD: Aplicar contexto de disidencias si existe
    if request.dissidenceContext:
        contextual_prompt = build_enhanced_dialectic_prompt(contextual_prompt, request.dissidenceContext)
    
    if request.initial_responses:
        initial_responses = {k: v['content'] for k, v in request.initial_responses.items()}
    else:
        initial_tasks = [call_ai_model_no_stream(m, contextual_prompt) for m in ['gemini', 'deepseek', 'claude']]
        results = await asyncio.gather(*initial_tasks)
        initial_responses = {'gemini': results[0], 'deepseek': results[1], 'claude': results[2]}

    # Construir prompts de crítica contextual
    critique_prompts = {}
    models_order = ['gemini', 'deepseek', 'claude']
    
    # CAMBIO CRÍTICO: Detectar si es la primera iteración del debate
    is_first_iteration = len(request.history) <= 1 or not any('Refinamiento dirigido' in turn.get('prompt', '') for turn in request.history)
    
    for model in models_order:
        context = "\n\n".join([f"**Respuesta de {m.title()}:**\n{r}" for m, r in initial_responses.items() if m != model])
        
        # CAMBIO CRÍTICO: Diferentes prompts según la iteración
        if is_first_iteration:
            # Primera iteración: Debate rico y crítica robusta
            base_critique = f"""**Tu Respuesta Inicial:**\n{initial_responses[model]}\n\n**Respuestas de Colegas:**\n{context}

**Tu Tarea:** Analiza críticamente las respuestas de tus colegas. Identifica fortalezas, debilidades y puntos ciegos en sus enfoques. Refina y mejora tu argumento incorporando nuevas perspectivas que enriquezcan el análisis."""
        else:
            # Segunda iteración en adelante: Convergencia dirigida
            base_critique = f"""**Tu Respuesta Inicial:**\n{initial_responses[model]}\n\n**Respuestas de Colegas:**\n{context}

**Tu Tarea:** Identifica elementos valiosos de sus respuestas que puedas integrar. Reformula tu análisis manteniendo consensos y abordando solo UNA diferencia crítica que consideres fundamental."""
        
        # Incluir contexto de refinamiento si existe
        if request.dissidenceContext and request.dissidenceContext.get('userRefinementPrompt'):
            user_guidance = request.dissidenceContext['userRefinementPrompt']
            base_critique += f"\n\n**Orientación Específica del Usuario:** {user_guidance}"
        
        critique_prompts[model] = base_critique
    
    critique_tasks = [call_ai_model_no_stream(m, critique_prompts[m]) for m in models_order]
    revised_results = await asyncio.gather(*critique_tasks)
    revised_responses = dict(zip(models_order, revised_results))
    
    # Síntesis final con contexto mejorado
    synthesis_context = "\n\n".join([f"**Argumento Revisado de {m.title()}:**\n{r}" for m, r in revised_responses.items()])
    synthesis_prompt = f"**Consulta (con historial):**\n{contextual_prompt}\n\n**Debate de Expertos:**\n{synthesis_context}\n\n**Tu Tarea:** Modera y crea un informe final unificado."
    
    # MEJORA: Añadir objetivos de consenso a la síntesis
    if request.dissidenceContext:
        confidence_level = request.dissidenceContext.get('confidenceLevel', 'balanced')
        target_consensus = request.dissidenceContext.get('targetConsensus', 70)
        synthesis_prompt += f"\n\n**Objetivo:** Crear una síntesis con nivel de confianza {confidence_level} (>{target_consensus}% consenso entre perspectivas)."
    
    # Manejar síntesis forzada
    if request.dissidenceContext and request.dissidenceContext.get('forcedSynthesis'):
        synthesis_prompt += "\n\n**INSTRUCCIÓN ESPECIAL:** Este es una síntesis forzada. Enfócate en los consensos existentes y presenta las diferencias restantes como perspectivas complementarias valiosas, no como conflictos a resolver."
    
    final_synthesis = await call_ai_model_no_stream('gemini', synthesis_prompt)

    return {
        "revised": revised_responses, 
        "synthesis": final_synthesis, 
        "initial": initial_responses,
        "dissidenceContext": request.dissidenceContext  # Retornar contexto para referencia
    }

@app.get("/")
async def root():
    return {"message": "Crisalia API v6.0 - Con Mejoras Dialécticas"}

@app.get("/health")
async def health_check():
    """Endpoint de salud para monitoreo"""
    return {
        "status": "healthy",
        "version": "6.0",
        "features": ["dialectic_enhancements", "extended_timeout", "dissidence_analysis"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

