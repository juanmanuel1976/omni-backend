import asyncio
import httpx
import os
import json

API_BASE_URL = os.environ.get("API_BASE_URL", "https://omni-backend-dev.onrender.com")

# --- EL DOCUMENTO TRAMPA (Contradicción Interna) ---
CONTRATO_FALSO = """
CONTRATO DE ARRENDAMIENTO COMERCIAL
... [Inicio del documento] ...
Cláusula 4 (De la Rescisión Anticipada): El arrendatario podrá rescindir el presente contrato a partir del sexto (6) mes de vigencia sin penalidad alguna, debiendo únicamente notificar de manera fehaciente con treinta (30) días de anticipación.
... [Siguen 15 párrafos de relleno legal irrelevante sobre pintura, expensas, ruidos molestos, horarios comerciales, seguros de incendio, limpieza de veredas, etc.] ...
(Relleno: El locatario debe mantener la pintura en buen estado. Queda prohibido ingresar con animales peligrosos. Las expensas extraordinarias corresponden al propietario...)
(Relleno: El seguro contra terceros es obligatorio y debe endosarse a favor del locador. El destino del local será únicamente comercial...)
... [Fin del relleno] ...
Cláusula 28 (Disposiciones Especiales y Penalidades): No obstante cualquier disposición anterior, toda rescisión anticipada que ocurra antes del primer año (12 meses) de vigencia devengará automáticamente una multa equivalente a dos (2) meses de alquiler a favor del arrendador, sin excepciones ni requerimiento de intimación previa.
... [Fin del documento] ...
"""

PREGUNTA_USUARIO = "El inquilino quiere rescindir el contrato en el mes 8 y ya envió la notificación hace 30 días. Según estrictamente este contrato, ¿debe pagar alguna multa?"

PROMPT_COMPLETO = f"**Contenido Completo de los Documento(s) para tu Análisis:**\n---\n{CONTRATO_FALSO}\n---\n\n**Consulta del Usuario:** {PREGUNTA_USUARIO}"

async def run_internal_contradiction_test():
    async with httpx.AsyncClient(timeout=300.0) as client:
        print("⚖️  INICIANDO BENCHMARK V3: DETECCIÓN DE CONTRADICCIÓN INTERNA\n")
        
        # 1. PRUEBA BASELINE
        print("🤖 Consultando a Gemini (Modelo Individual)...")
        res_base = await client.post(f"{API_BASE_URL}/api/generate-initial", json={
            "prompt": PROMPT_COMPLETO,
            "model": "gemini"
        })
        if res_base.status_code != 200:
            print(f"❌ Falló Gemini. Código HTTP: {res_base.status_code}\nDetalle: {res_base.text}")
            return
            
        respuesta_gemini = res_base.json().get("response", "")
        print("✅ Gemini respondió.\n")

        # 2. PRUEBA CRISALIA
        print("🧠 Consultando a Crisalia (Debate Multi-Agente)...")
        res_crisalia = await client.post(f"{API_BASE_URL}/api/debate", json={
            "prompt": PROMPT_COMPLETO,
            "isDocument": True 
        })
        
        # LA LUPA: Revisamos qué devolvió el servidor
        if res_crisalia.status_code != 200:
            print(f"❌ Falló Crisalia. Código HTTP: {res_crisalia.status_code}\nDetalle exacto del error del servidor:\n{res_crisalia.text}")
            return
            
        respuesta_crisalia = res_crisalia.json().get("synthesis", "")
        print("✅ Crisalia sintetizó.\n")

        # 3. EL JUEZ CIEGO
        print("🧑‍⚖️ Enviando respuestas al Juez Ciego (Claude)...")
        prompt_juez = f"""Evalúa estas dos respuestas de IA frente a un contrato que tiene una trampa (una contradicción evidente entre la Cláusula 4 y la Cláusula 28).

**Contrato original:**
{CONTRATO_FALSO}

**Pregunta del usuario:** {PREGUNTA_USUARIO}

**Respuesta A:** {respuesta_gemini}
**Respuesta B:** {respuesta_crisalia}

**Instrucciones de Evaluación:**
1. Un buen analista legal debe notar que el contrato se contradice y advertir al usuario sobre el riesgo de litigio.
2. Un mal analista simplemente leerá la primera cláusula o la última y dará una respuesta afirmativa o negativa simple, ignorando el conflicto.
3. Evalúa qué respuesta detectó mejor la contradicción interna y aconsejó mejor al usuario basándose ESTRICTAMENTE en el texto.

Responde en JSON:
{{
    "analisis_A": "Por qué la respuesta A es buena o mala",
    "analisis_B": "Por qué la respuesta B es buena o mala",
    "ganador": "A o B o Empate",
    "detecto_contradiccion_A": bool,
    "detecto_contradiccion_B": bool
}}"""

        res_juez = await client.post(f"{API_BASE_URL}/api/generate-initial", json={
            "prompt": prompt_juez,
            "model": "claude"
        })
        
        if res_juez.status_code != 200:
            print(f"❌ Falló el Juez. Código HTTP: {res_juez.status_code}\nDetalle: {res_juez.text}")
            return
            
        veredicto = res_juez.json().get("response", "")
        print("\n🏆 VEREDICTO FINAL DEL JUEZ:\n")
        print(veredicto)

if __name__ == "__main__":
    asyncio.run(run_internal_contradiction_test())
