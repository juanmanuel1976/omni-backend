import logging
from google.cloud import vision
import fitz  # PyMuPDF
import io
import os
from google.api_core.client_options import ClientOptions

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Procesador OCR para PDFs escaneados usando Google Cloud Vision"""
    
    def __init__(self):
        # NO crear el cliente aquí, solo guardar la API key
        self.api_key = os.environ.get("GOOGLE_VISION_API_KEY")
        self.client = None
    
    def _get_client(self):
        """Crear cliente lazy (solo cuando se necesita)"""
        if self.client is None and self.api_key:
            try:
                # Configurar client options con API key
                client_options = ClientOptions(api_key=self.api_key)
                self.client = vision.ImageAnnotatorClient(client_options=client_options)
                logger.info("Google Vision API client inicializado correctamente")
            except Exception as e:
                logger.error(f"Error inicializando Vision API client: {e}")
                self.client = None
        return self.client
    
    def is_pdf_scanned(self, pdf_bytes: bytes) -> bool:
    """
    Detecta si un PDF es escaneado o híbrido (mezcla de nativo + escaneado).
    Optimizado para expedientes judiciales.
    """
    try:
        import pypdf
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        
        total_pages = len(pdf_reader.pages)
        
        # Muestreo inteligente: revisar más páginas distribuidas
        if total_pages <= 10:
            # PDFs pequeños: revisar todas
            pages_to_test = list(range(total_pages))
        elif total_pages <= 50:
            # PDFs medianos: revisar 10 páginas distribuidas
            pages_to_test = [0, 1, 2] + list(range(10, total_pages, max(1, total_pages // 7)))[:7]
        else:
            # PDFs grandes: revisar 15 páginas distribuidas (inicio, medio, final)
            step = total_pages // 15
            pages_to_test = list(range(0, total_pages, max(1, step)))[:15]
        
        pages_with_text = 0
        pages_without_text = 0
        total_chars = 0
        
        for page_num in pages_to_test:
            if page_num >= total_pages:
                continue
                
            page_text = pdf_reader.pages[page_num].extract_text() or ""
            char_count = len(page_text.strip())
            total_chars += char_count
            
            # Página con poco texto (<50 chars) = probablemente escaneada
            if char_count < 50:
                pages_without_text += 1
            else:
                pages_with_text += 1
        
        pages_tested = len(pages_to_test)
        avg_chars_per_page = total_chars / pages_tested if pages_tested > 0 else 0
        scanned_ratio = pages_without_text / pages_tested if pages_tested > 0 else 0
        
        # CRITERIOS MEJORADOS:
        # 1. Si >30% de páginas muestreadas no tienen texto → ES ESCANEADO
        # 2. O si promedio <100 chars/página → ES ESCANEADO
        is_scanned = (scanned_ratio > 0.3) or (avg_chars_per_page < 100)
        
        logger.info(f"PDF análisis: {pages_tested} páginas testeadas de {total_pages} totales")
        logger.info(f"Páginas con texto: {pages_with_text}, sin texto: {pages_without_text}")
        logger.info(f"Promedio: {avg_chars_per_page:.0f} chars/página, ratio escaneado: {scanned_ratio:.2%}")
        logger.info(f"Conclusión: {'ESCANEADO/HÍBRIDO - Usando OCR' if is_scanned else 'NATIVO - Sin OCR'}")
        
        return is_scanned
        
    except Exception as e:
        logger.error(f"Error detectando PDF escaneado: {e}")
        # En caso de error, asumir que SÍ es escaneado (mejor OCR innecesario que perder datos)
        return True
    
    async def extract_text_with_ocr(self, pdf_bytes: bytes, max_pages: int = None) -> str:
        """
        Extrae texto de PDF escaneado usando OCR.
        Usa PyMuPDF para convertir PDF a imágenes (NO necesita poppler).
        """
        client = self._get_client()
        
        if not client:
            logger.warning("Google Vision API no configurada, saltando OCR")
            return ""
        
        try:
            # Abrir PDF con PyMuPDF
            logger.info("Convirtiendo PDF a imágenes para OCR con PyMuPDF...")
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            total_pages = len(pdf_document)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            logger.info(f"Procesando {pages_to_process} páginas con OCR...")
            
            all_text = []
            
            for page_num in range(pages_to_process):
                # Convertir página a imagen
                page = pdf_document[page_num]
                
                # Renderizar página a imagen (300 DPI para mejor calidad OCR)
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # Convertir a bytes JPEG
                img_bytes = pix.tobytes("jpeg")
                
                # Llamar a Vision API
                vision_image = vision.Image(content=img_bytes)
                response = client.text_detection(image=vision_image)
                
                if response.text_annotations:
                    page_text = response.text_annotations[0].description
                    all_text.append(f"\n--- Página {page_num + 1} ---\n{page_text}")
                
                # Logging de progreso
                if (page_num + 1) % 10 == 0:
                    logger.info(f"OCR: {page_num + 1}/{pages_to_process} páginas procesadas")
            
            pdf_document.close()
            
            extracted_text = "\n".join(all_text)
            logger.info(f"OCR completado: {len(extracted_text)} caracteres extraídos de {pages_to_process} páginas")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error en OCR: {e}")
            return ""

# Instancia global
ocr_processor = OCRProcessor()
