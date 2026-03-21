import subprocess
import httpx
import sys
import json
import argparse
import os

# URL de tu Crisalia en Producción
API_URL = "https://omni-backend-b9p2.onrender.com/api/validate-change"

# Límites de tokens (caracteres aprox.) para controlar costos
MAX_CONTEXT_CHARS = 2000   # ~500 tokens
MAX_INTENCION_CHARS = 800  # ~200 tokens

CONTEXTO_FILE = os.path.join(os.path.dirname(__file__), "CONTEXTO.md")


def get_git_diff():
    """Obtiene los cambios actuales no comiteados."""
    try:
        result = subprocess.run(['git', 'diff', 'HEAD'], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception:
        print("Error: Asegúrate de estar en un repositorio Git y tener cambios sin comitear.")
        sys.exit(1)


def load_contexto():
    """Lee CONTEXTO.md y retorna su contenido truncado al límite de caracteres."""
    if not os.path.exists(CONTEXTO_FILE):
        return ""
    with open(CONTEXTO_FILE, encoding="utf-8") as f:
        contenido = f.read()
    if len(contenido) > MAX_CONTEXT_CHARS:
        contenido = contenido[:MAX_CONTEXT_CHARS] + "\n... [contexto truncado]"
    return contenido


def build_descripcion(intencion, contexto):
    """Construye el campo 'descripcion' que reciben los jueces, con contexto e intención."""
    partes = []

    if contexto:
        partes.append("=== CONTEXTO DEL PROYECTO ===\n" + contexto)

    if intencion:
        intencion_truncada = intencion[:MAX_INTENCION_CHARS]
        partes.append("=== INTENCIÓN DEL CAMBIO ===\n" + intencion_truncada)

    partes.append("=== REVISIÓN ===\nRevisión automática de Claude Code pre-commit.")

    return "\n\n".join(partes)


def parse_args():
    parser = argparse.ArgumentParser(description="Tribunal Dialéctico de Crisalia — validador de cambios")
    parser.add_argument(
        "-m", "--mensaje",
        type=str,
        default="",
        help='Intención del cambio. Ej: -m "OBJETIVO: crear dashboard de gastos LLM | MOTIVACION: controlar costos | IMPACTO: omni_app.py"'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("⏳ Ejecutando git diff...")
    diff_content = get_git_diff()

    if not diff_content.strip():
        print("✅ No hay cambios en el código para validar.")
        sys.exit(0)

    contexto = load_contexto()
    descripcion = build_descripcion(args.mensaje, contexto)

    if args.mensaje:
        print(f"📋 Intención: {args.mensaje[:80]}{'...' if len(args.mensaje) > 80 else ''}")
    if contexto:
        print(f"📂 Contexto del proyecto cargado ({len(contexto)} chars)")

    print("🤖 Solicitando Tribunal Dialéctico de Crisalia (Gemini + DeepSeek + Claude)...")
    print("⏳ Esto puede demorar unos minutos. Por favor, espera.")

    payload = {
        "archivo": "Múltiples (Git Diff)",
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
