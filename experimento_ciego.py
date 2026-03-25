# -*- coding: utf-8 -*-
"""
Experimento ciego: Perplexity sonar-pro vs Crisalia vs Crisalia+Crucible
Juez: GPT-4o via /api/gpt-compare
4 prompts: 3 sin búsqueda web, 1 con búsqueda web
"""
import urllib.request, json, ssl, sys, time, random
sys.stdout.reconfigure(encoding='utf-8')
ctx = ssl.create_default_context()

import os
PERPLEXITY_KEY = os.environ.get('PERPLEXITY_KEY', '')
DEV_URL        = os.environ.get('DEV_URL', 'https://omni-backend-dev.onrender.com')
OWNER_KEY      = os.environ.get('OWNER_KEY', '')

# ─── PROMPTS ─────────────────────────────────────────────────────────────────
PROMPTS = [
    {
        "id": 1,
        "web": False,
        "categoria": "Estrategia de negocio",
        "texto": "Una empresa B2B de software tiene una tasa de churn del 8% mensual. "
                 "El producto funciona bien técnicamente pero los clientes se van. "
                 "¿Cuáles son las 3 causas más probables y qué acciones concretas tomarías en los próximos 30 días?"
    },
    {
        "id": 2,
        "web": False,
        "categoria": "Dilema ético",
        "texto": "Una empresa de IA descubre que su modelo de contratación automatizado rechaza "
                 "sistemáticamente candidatos de ciertos barrios, aunque no usa raza ni etnia como variable explícita. "
                 "El modelo tiene 94% de precisión y mejora la performance del equipo. "
                 "¿Qué debe hacer la empresa? ¿Apagarlo, corregirlo, o seguir usándolo mientras se corrige?"
    },
    {
        "id": 3,
        "web": False,
        "categoria": "Análisis técnico",
        "texto": "Un equipo de ingeniería tiene que decidir entre migrar su monolito de 5 años "
                 "a microservicios o hacer un 'modular monolith'. El equipo tiene 8 personas, "
                 "el sistema procesa 50k transacciones/día, y tienen 6 meses de runway. "
                 "¿Cuál elegís y por qué?"
    },
    {
        "id": 4,
        "web": True,
        "categoria": "Actualidad con búsqueda web",
        "texto": "w. ¿Cuál es el estado actual de la regulación de IA en la Unión Europea "
                 "y qué deben hacer las startups de IA latinoamericanas que quieran vender allí en 2026?"
    },
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def post(url, payload, headers, timeout=240):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req  = urllib.request.Request(url, data=body, headers=headers, method='POST')
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))

def call_perplexity(prompt_text):
    payload = {
        'model': 'sonar-pro',
        'messages': [{'role': 'user', 'content': prompt_text}],
        'max_tokens': 1200, 'temperature': 0.7,
    }
    headers = {'Authorization': f'Bearer {PERPLEXITY_KEY}', 'Content-Type': 'application/json'}
    data = post('https://api.perplexity.ai/chat/completions', payload, headers, timeout=90)
    return data['choices'][0]['message']['content']

def call_crisalia(prompt_text):
    payload = {'prompt': prompt_text, 'lang': 'es'}
    headers = {'Content-Type': 'application/json; charset=utf-8', 'X-Owner-Key': OWNER_KEY}
    data = post(f'{DEV_URL}/api/debate', payload, headers, timeout=300)
    return data.get('synthesis', '') or data.get('sintesis', '')

def call_crucible(sintesis, contexto):
    payload = {'sintesis': sintesis, 'contexto': contexto, 'archivo': 'crucible_experimento'}
    headers = {'Content-Type': 'application/json; charset=utf-8', 'X-Owner-Key': OWNER_KEY}
    data = post(f'{DEV_URL}/api/crucible', payload, headers)
    return data.get('sintesis_final', ''), data.get('veredicto_final', ''), data.get('ataque', '')

def call_gpt_compare(pregunta, resp_a, resp_b):
    payload  = {'pregunta': pregunta, 'respuesta_a': resp_a, 'respuesta_b': resp_b}
    headers  = {'Content-Type': 'application/json; charset=utf-8', 'X-Owner-Key': OWNER_KEY}
    return post(f'{DEV_URL}/api/gpt-compare', payload, headers, timeout=60)

# ─── EXPERIMENTO ─────────────────────────────────────────────────────────────
resultados = []

for p in PROMPTS:
    print(f'\n{"="*70}')
    print(f'PROMPT {p["id"]} — {p["categoria"]}')
    print(f'{"="*70}')
    print(f'{p["texto"][:120]}...\n')

    # PASO 1: Perplexity
    print('  [1/4] Perplexity sonar-pro...', end=' ', flush=True)
    t = time.time()
    resp_perplexity = call_perplexity(p['texto'])
    print(f'OK ({int(time.time()-t)}s, {len(resp_perplexity)} chars)')

    # PASO 2: Crisalia
    print('  [2/4] Crisalia (Tribunal)...', end=' ', flush=True)
    t = time.time()
    # Para web: quitar el "w." del texto que va a Crisalia
    texto_crisalia = p['texto'].replace('w. ', '', 1)
    resp_crisalia = call_crisalia(texto_crisalia)
    print(f'OK ({int(time.time()-t)}s, {len(resp_crisalia)} chars)')

    # PASO 3: Crucible sobre Crisalia
    print('  [3/4] Crucible (Abogado del Diablo)...', end=' ', flush=True)
    t = time.time()
    resp_crucible, veredicto_crucible, ataque_crucible = call_crucible(resp_crisalia, texto_crisalia)
    print(f'OK ({int(time.time()-t)}s) — Veredicto: {veredicto_crucible}')

    # PASO 4a: GPT juzga Perplexity vs Crisalia (orden aleatorio para ceguera)
    print('  [4/4] GPT-4o juzga...', end=' ', flush=True)
    t = time.time()
    orden_flip = random.random() > 0.5
    if orden_flip:
        juicio_pc = call_gpt_compare(texto_crisalia, resp_crisalia, resp_perplexity)
        # corregir etiquetas: A=Crisalia, B=Perplexity → mapear al revés
        ganador_pc_raw = juicio_pc.get('ganador','')
        ganador_pc = 'Crisalia' if ganador_pc_raw == 'A' else ('Perplexity' if ganador_pc_raw == 'B' else 'empate')
        score_crisalia_pc  = juicio_pc.get('A', {}).get('total', 0)
        score_perplexity_pc = juicio_pc.get('B', {}).get('total', 0)
    else:
        juicio_pc = call_gpt_compare(texto_crisalia, resp_perplexity, resp_crisalia)
        ganador_pc_raw = juicio_pc.get('ganador','')
        ganador_pc = 'Perplexity' if ganador_pc_raw == 'A' else ('Crisalia' if ganador_pc_raw == 'B' else 'empate')
        score_perplexity_pc = juicio_pc.get('A', {}).get('total', 0)
        score_crisalia_pc   = juicio_pc.get('B', {}).get('total', 0)

    # PASO 4b: GPT juzga Perplexity vs Crucible
    if orden_flip:
        juicio_pcu = call_gpt_compare(texto_crisalia, resp_crucible, resp_perplexity)
        ganador_pcu_raw = juicio_pcu.get('ganador','')
        ganador_pcu = 'Crucible' if ganador_pcu_raw == 'A' else ('Perplexity' if ganador_pcu_raw == 'B' else 'empate')
        score_crucible_pcu   = juicio_pcu.get('A', {}).get('total', 0)
        score_perplexity_pcu = juicio_pcu.get('B', {}).get('total', 0)
    else:
        juicio_pcu = call_gpt_compare(texto_crisalia, resp_perplexity, resp_crucible)
        ganador_pcu_raw = juicio_pcu.get('ganador','')
        ganador_pcu = 'Perplexity' if ganador_pcu_raw == 'A' else ('Crucible' if ganador_pcu_raw == 'B' else 'empate')
        score_perplexity_pcu = juicio_pcu.get('A', {}).get('total', 0)
        score_crucible_pcu   = juicio_pcu.get('B', {}).get('total', 0)

    print(f'OK ({int(time.time()-t)}s)')
    print(f'  Perplexity vs Crisalia  → {ganador_pc} gana ({score_perplexity_pc} vs {score_crisalia_pc})')
    print(f'  Perplexity vs Crucible  → {ganador_pcu} gana ({score_perplexity_pcu} vs {score_crucible_pcu})')

    resultados.append({
        'id': p['id'],
        'categoria': p['categoria'],
        'web': p['web'],
        'prompt': p['texto'],
        'resp_perplexity': resp_perplexity,
        'resp_crisalia': resp_crisalia,
        'resp_crucible': resp_crucible,
        'ataque_crucible': ataque_crucible,
        'veredicto_crucible': veredicto_crucible,
        'juicio_pc': juicio_pc,
        'juicio_pcu': juicio_pcu,
        'score_perplexity': score_perplexity_pc,
        'score_crisalia': score_crisalia_pc,
        'score_crucible': score_crucible_pcu,
        'ganador_pc': ganador_pc,
        'ganador_pcu': ganador_pcu,
    })

# ─── GUARDAR ─────────────────────────────────────────────────────────────────
out = 'C:/Users/LimaVictor/Desktop/CLAUDE&CRISALIA/_versions/experimento_ciego.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print(f'\n\nGuardado: {out}')

# ─── RESUMEN ─────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('RESUMEN FINAL')
print('='*70)
for r in resultados:
    print(f'Prompt {r["id"]} ({r["categoria"]}):')
    print(f'  Perplexity {r["score_perplexity"]} | Crisalia {r["score_crisalia"]} | Crucible {r["score_crucible"]}')
    print(f'  Perplexity vs Crisalia → {r["ganador_pc"]}')
    print(f'  Perplexity vs Crucible → {r["ganador_pcu"]}')

avg_p  = sum(r['score_perplexity'] for r in resultados) / len(resultados)
avg_c  = sum(r['score_crisalia']   for r in resultados) / len(resultados)
avg_cu = sum(r['score_crucible']   for r in resultados) / len(resultados)
print(f'\nPROMEDIO TOTAL:')
print(f'  Perplexity : {avg_p:.1f}')
print(f'  Crisalia   : {avg_c:.1f}')
print(f'  Crucible   : {avg_cu:.1f}')
