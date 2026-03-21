import subprocess
import httpx
import sys
import argparse
import os
import re

# URL de tu Crisalia en Producción
API_URL = "https://omni-backend-b9p2.onrender.com/api/validate-change"

# Límites de caracteres para controlar costos (~tokens)
MAX_CONTEXT_CHARS = 2000   # ~500 tokens
MAX_INTENCION_CHARS = 800  # ~200 tokens

CONTEXTO_FILE = os.path.join(os.path.dirname(__file__), "CONTEXTO.md")

# Fase 2: mapa de patrones de archivos → tags relevantes de CONTEXTO.md
FILE_TAG_MAP = [
    (r"omni_app\.py",           ["ARQUITECTURA", "RESTRICCIONES", "DECISIONES_IRREVOCABLES"]),
    (r"app\.html",              ["FASE", "NEGOCIO", "RESTRICCIONES"]),
    (r"index\.html",            ["FASE", "NEGOCIO"]),
    (r"crisalia_bridge\.py",    ["ARQUITECTURA", "DECISIONES_IRREVOCABLES"]),
    (r"stripe_handler\.py",     ["FASE", "NEGOCIO", "RESTRICCIONES"]),
    (r"rag_manager\.py",        ["ARQUITECTURA"]),
    (r"ocr_processor\.py",      ["ARQUITECTURA"]),
    (r"CLAUDE\.md",             ["DECISIONES_IRREVOCABLES"]),
    (r"CONTEXTO\.md",           ["PROYECTO", "FASE", "ARQUITECTURA", "DECISIONES_IRREVOCABLES", "RESTRICCIONES", "NEGOCIO"]),
    (r"requirements\.txt",      ["ARQUITECTURA"]),
]
# Si un archivo no matchea ningún patrón, se incluyen estos tags por defecto
DEFAULT_TAGS = ["PROYECTO", "FASE"]


def get_git_diff():
    """Obtiene los cambios actuales no comiteados."""
    try:
        result = subprocess.run(['git', 'diff', 'HEAD'], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception:
        print("Error: Asegúrate de estar en un repositorio Git y tener cambios sin comitear.")
        sys.exit(1)


def extract_files_from_diff(diff):
    """Extrae la lista de archivos tocados a partir del git diff."""
    files = []
    for line in diff.splitlines():
        # Las líneas de diff tienen el formato: diff --git a/archivo b/archivo
        if line.startswith("diff --git "):
            match = re.search(r'diff --git a/(.+?) b/', line)
            if match:
                files.append(match.group(1))
    return files


def get_relevant_tags(files):
    """Dado un listado de archivos, retorna los tags únicos relevantes."""
    tags = set()
    for filepath in files:
        filename = os.path.basename(filepath)
        matched = False
        for pattern, file_tags in FILE_TAG_MAP:
            if re.search(pattern, filename, re.IGNORECASE):
                tags.update(file_tags)
                matched = True
        if not matched:
            tags.update(DEFAULT_TAGS)
    return tags


def load_contexto_filtrado(tags):
    """
    Lee CONTEXTO.md y retorna solo las secciones cuyos tags son relevantes.
    Cada sección comienza con una línea [TAG].
    """
    if not os.path.exists(CONTEXTO_FILE):
        return ""

    with open(CONTEXTO_FILE, encoding="utf-8") as f:
        lineas = f.readlines()

    secciones = []
    seccion_actual = []
    tag_actual = None

    for linea in lineas:
        tag_match = re.match(r'^\[([A-Z_]+)\]', linea.strip())
        if tag_match:
            # Guardar la sección anterior si su tag era relevante
            if tag_actual and tag_actual in tags and seccion_actual:
                secciones.append("".join(seccion_actual).strip())
            tag_actual = tag_match.group(1)
            seccion_actual = [linea]
        else:
            if tag_actual:
                seccion_actual.append(linea)

    # Última sección
    if tag_actual and tag_actual in tags and seccion_actual:
        secciones.append("".join(seccion_actual).strip())

    contexto = "\n\n".join(secciones)

    if len(contexto) > MAX_CONTEXT_CHARS:
        contexto = contexto[:MAX_CONTEXT_CHARS] + "\n... [contexto truncado]"

    return contexto


def build_descripcion(intencion, contexto, tags_usados, archivos_tocados):
    """Construye el campo 'descripcion' que reciben los jueces."""
    partes = []

    if contexto:
        tags_str = ", ".join(sorted(tags_usados))
        partes.append(f"=== CONTEXTO DEL PROYECTO (secciones: {tags_str}) ===\n{contexto}")

    if intencion:
        partes.append("=== INTENCIÓN DEL CAMBIO ===\n" + intencion[:MAX_INTENCION_CHARS])

    if archivos_tocados:
        partes.append("=== ARCHIVOS MODIFICADOS ===\n" + "\n".join(f"- {f}" for f in archivos_tocados))

    partes.append("=== REVISIÓN ===\nRevisión automática de Claude Code pre-commit.")

    return "\n\n".join(partes)


def parse_args():
    parser = argparse.ArgumentParser(description="Tribunal Dialéctico de Crisalia — validador de cambios")
    parser.add_argument(
        "-m", "--mensaje",
        type=str,
        default="",
        help='Intención del cambio. Ej: -m "OBJETIVO: crear dashboard | MOTIVACION: controlar costos | IMPACTO: omni_app.py"'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("⏳ Ejecutando git diff...")
    diff_content = get_git_diff()

    if not diff_content.strip():
        print("✅ No hay cambios en el código para validar.")
        sys.exit(0)

    # Fase 2: detectar archivos y filtrar contexto
    archivos_tocados = extract_files_from_diff(diff_content)
    tags_relevantes = get_relevant_tags(archivos_tocados)
    contexto = load_contexto_filtrado(tags_relevantes)

    descripcion = build_descripcion(args.mensaje, contexto, tags_relevantes, archivos_tocados)

    if archivos_tocados:
        print(f"📁 Archivos en el diff: {', '.join(archivos_tocados)}")
    if tags_relevantes:
        print(f"🏷️  Tags de contexto seleccionados: {', '.join(sorted(tags_relevantes))}")
    if contexto:
        print(f"📂 Contexto filtrado cargado ({len(contexto)} chars)")
    if args.mensaje:
        print(f"📋 Intención: {args.mensaje[:80]}{'...' if len(args.mensaje) > 80 else ''}")

    print("🤖 Solicitando Tribunal Dialéctico de Crisalia (Gemini + DeepSeek + Claude)...")
    print("⏳ Esto puede demorar unos minutos. Por favor, espera.")

    payload = {
        "archivo": ", ".join(archivos_tocados) if archivos_tocados else "Múltiples (Git Diff)",
        "descripcion": descripcion,
        "codigo_original": "Ver Diff",
        "codigo_propuesto": diff_content,
        "tipo": "code"
    }

    try:
        response = httpx.post(API_URL, json=payload, timeout=300.0)
        response.raise_for_status()

        data = response.json()

        print("\n" + "="*50)
        print(f"⚖️ VEREDICTO DE CRISALIA: {data.get('veredicto', 'DESCONOCIDO')}")
        print("="*50)
        print(f"📊 Consenso de IAs: {data.get('consenso_score', 0)}%")
        print(f"📝 Síntesis: {data.get('sintesis', '')}")

        if data.get('riesgos'):
            print("\n⚠️ RIESGOS DETECTADOS:")
            for riesgo in data['riesgos']:
                print(f"  - {riesgo}")

        if data.get('sugerencias'):
            print("\n💡 SUGERENCIAS DE CORRECCIÓN:")
            for sug in data['sugerencias']:
                print(f"  - {sug}")
        print("="*50 + "\n")

    except httpx.ReadTimeout:
        print("⚠️ TIMEOUT: Crisalia tardó demasiado. (Falla en Modo Permisivo)")
    except Exception as e:
        print(f"❌ ERROR de conexión con Crisalia: {e}")


if __name__ == "__main__":
    main()
