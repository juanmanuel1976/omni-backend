import logging
from google.cloud import vision
from pdf2image import convert_from_bytes
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
        Detecta si un PDF es escaneado (tiene poco texto extraíble).
        """
        try:
            import pypdf
            pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            
            # Testear primeras 3 páginas
            total_text = ""
            pages_to_test = min(3, len(pdf_reader.pages))
            
            for i in range(pages_to_test):
                page_text = pdf_reader.pages[i].extract_text() or ""
                total_text += page_text
            
            # Si tiene <100 chars por página → es escaneado
            avg_chars_per_page = len(total_text) / pages_to_test if pages_to_test > 0 else 0
            is_scanned = avg_chars_per_page < 100
            
            logger.info(f"PDF escaneado: {is_scanned} ({avg_chars_per_page:.0f} chars/página)")
            return is_scanned
            
        except Exception as e:
            logger.error(f"Error detectando PDF escaneado: {e}")
            return False
    
    async def extract_text_with_ocr(self, pdf_bytes: bytes, max_pages: int = None) -> str:
        """
        Extrae texto de PDF escaneado usando OCR.
        """
        client = self._get_client()
        
        if not client:
            logger.warning("Google Vision API no configurada, saltando OCR")
            return ""
        
        try:
            # Convertir PDF a imágenes
            logger.info("Convirtiendo PDF a imágenes para OCR...")
            images = convert_from_bytes(pdf_bytes, dpi=200, fmt='jpeg')
            
            if max_pages:
                images = images[:max_pages]
            
            logger.info(f"Procesando {len(images)} páginas con OCR...")
            
            all_text = []
            
            for i, image in enumerate(images):
                # Convertir imagen a bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                # Llamar a Vision API
                vision_image = vision.Image(content=img_bytes)
                response = client.text_detection(image=vision_image)
                
                if response.text_annotations:
                    page_text = response.text_annotations[0].description
                    all_text.append(f"\n--- Página {i+1} ---\n{page_text}")
                
                # Logging de progreso
                if (i + 1) % 10 == 0:
                    logger.info(f"OCR: {i+1}/{len(images)} páginas procesadas")
            
            extracted_text = "\n".join(all_text)
            logger.info(f"OCR completado: {len(extracted_text)} caracteres extraídos")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error en OCR: {e}")
            return ""

# Instancia global
ocr_processor = OCRProcessor()
