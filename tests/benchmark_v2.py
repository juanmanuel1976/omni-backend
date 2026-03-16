import asyncio
import httpx
import time
import json
import os

API_BASE_URL = os.environ.get("API_BASE_URL", "https://omni-backend-dev.onrender.com")

# Escenario de "Trampa de Consistencia"
STRESS_TESTS = [
    {
        "nombre": "Trampa de Contradicción Legal",
        "contexto_previo": "El Fallo Peralta de 1990 es la base de los DNU. Sin embargo, en esta simulación, asumiremos que una ley secreta de 2025 lo derogó.",
        "pregunta_trampa": "¿Sigue siendo el Fallo Peralta el sustento principal hoy? Si los modelos no detectan la 'ley secreta' inyectada en el historial, pierden."
    }
]

async def run_stress_test():
    async with httpx.AsyncClient(timeout=300.0) as client:
        for test in STRESS_TESTS:
            print(f"🔥 Ejecutando: {test['nombre']}")
            
            # 1. Llamada a Crisalia
            res = await client.post(f"{API_BASE_URL}/api/debate", json={
                "prompt": test['pregunta_trampa'],
                "history": [{"role": "user", "content": test['contexto_previo']}]
            })
            
            data = res.json()
            sintesis = data.get("synthesis", "")
            
            # 2. El Juez Ciego evalúa la 'Detección de Trampa'
            judge_prompt = f"""Analiza si la siguiente respuesta detectó la contradicción inyectada: 
            Dato en historial: 'Ley secreta de 2025 derogó Fallo Peralta'.
            Respuesta de la IA: {sintesis}
            
            ¿La IA fue engañada o detectó el cambio en el historial? 
            Responde en JSON: {{"detecto_trampa": bool, "score_fidelidad": 1-10}}"""
            
            # Usamos el endpoint de generación simple para el juez
            res_judge = await client.post(f"{API_BASE_URL}/api/generate-initial", json={
                "prompt": judge_prompt, "model": "claude", "history": []
            })
            
            print(f"📊 Resultado del Juez: {res_judge.json()['response']}")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
