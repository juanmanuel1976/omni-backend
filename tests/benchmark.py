import asyncio
import httpx
import time
import json
import os
from datetime import datetime

# Configuración
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000") # Cambia a https://dev.crisalia.io para testear en prod
OUTPUT_DIR = "benchmarks_reports"

# Consultas complejas para estresar el sistema (Enfocadas en escenarios crítico/político/legales)
QUERIES = [
    "Analiza la viabilidad constitucional y el impacto político de gobernar mediante Decretos de Necesidad y Urgencia (DNU) frente a un Congreso fragmentado en Argentina.",
    "¿Cuáles son las implicancias legales y los vacíos regulatorios del uso de IA generativa para la redacción de sentencias judiciales?",
    "Diseña una estrategia de contención de crisis (PR) para un candidato político que acaba de sufrir una filtración de audios comprometedores a una semana de las elecciones."
]

async def evaluar_con_juez(sintesis, respuesta_individual):
    """Usa a Claude como juez ciego para comparar la síntesis vs la respuesta aislada."""
    prompt_juez = f"""Eres un juez imparcial evaluando respuestas de IA.
    Evalúa cuál respuesta es más completa, neutral y estratégicamente útil.
    
    Respuesta A (Modelo Individual):
    {respuesta_individual[:1500]}...
    
    Respuesta B (Síntesis Crisalia):
    {sintesis[:1500]}...
    
    Responde SOLO con un JSON válido con este formato:
    {{"ganador": "A o B", "razon_breve": "1 oración explicando por qué", "score_A": 1-10, "score_B": 1-10}}
    """
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            res = await client.post(
                f"{API_BASE_URL}/api/generate-initial",
                json={"prompt": prompt_juez, "history": [], "model": "claude"}
            )
            data = res.json()
            # Limpiar JSON por si el modelo devuelve markdown
            texto_limpio = data["response"].replace("```json", "").replace("```", "").strip()
            return json.loads(texto_limpio)
        except Exception as e:
            return {"ganador": "Error", "razon_breve": str(e), "score_A": 0, "score_B": 0}

async def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{OUTPUT_DIR}/benchmark_crisalia_{timestamp}.md"
    
    print(f"🚀 Iniciando Crisalia Benchmark Suite...")
    print(f"📡 Target API: {API_BASE_URL}\n")
    
    resultados = []
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        for i, query in enumerate(QUERIES, 1):
            print(f"▶️ Test {i}/{len(QUERIES)}: {query[:50]}...")
            
            # 1. Medir Debate Dialéctico Completo
            start_time = time.time()
            res_debate = await client.post(f"{API_BASE_URL}/api/debate", json={"prompt": query, "history": []})
            latency = time.time() - start_time
            
            if res_debate.status_code != 200:
                print(f"❌ Error en debate: {res_debate.text}")
                continue
                
            data_debate = res_debate.json()
            respuestas_iniciales = data_debate.get("initial", {})
            sintesis_final = data_debate.get("synthesis", "")
            
            # 2. Medir Consenso Semántico Inicial
            res_consenso = await client.post(
                f"{API_BASE_URL}/api/semantic-consensus",
                json={"responses": respuestas_iniciales}
            )
            consenso = res_consenso.json().get("consensus_score", 0) if res_consenso.status_code == 200 else "Error"
            
            # 3. Evaluación del Juez Ciego (Crisalia vs Gemini en solitario)
            evaluacion = await evaluar_con_juez(sintesis_final, respuestas_iniciales.get("gemini", ""))
            
            resultados.append({
                "query": query,
                "latencia": round(latency, 2),
                "consenso_inicial": consenso,
                "evaluacion": evaluacion
            })
            print(f"   ✓ Latencia: {latency:.2f}s | Consenso: {consenso}% | Ganador: {evaluacion.get('ganador')}")

    # 4. Generar Reporte Markdown
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("# 📊 Crisalia - Reporte Oficial de Benchmark\n\n")
        f.write(f"**Fecha de Ejecución:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Target API:** `{API_BASE_URL}`\n\n")
        
        for i, r in enumerate(resultados, 1):
            f.write(f"## Test {i}: {r['query']}\n")
            f.write(f"- **Tiempo de Ejecución:** {r['latencia']} segundos\n")
            f.write(f"- **Nivel de Consenso Base:** {r['consenso_inicial']}%\n")
            f.write("### Veredicto del Juez Ciego (Gemini vs Síntesis Crisalia)\n")
            f.write(f"- **Ganador:** **{'CRISALIA' if r['evaluacion']['ganador'] == 'B' else 'MODELO INDIVIDUAL'}**\n")
            f.write(f"- **Puntaje Crisalia:** {r['evaluacion'].get('score_B', 0)}/10\n")
            f.write(f"- **Puntaje Gemini Aislado:** {r['evaluacion'].get('score_A', 0)}/10\n")
            f.write(f"- **Razón:** {r['evaluacion'].get('razon_breve', 'N/A')}\n\n")
            f.write("---\n")
            
    print(f"\n✅ Benchmark completado. Reporte guardado en: {report_filename}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
