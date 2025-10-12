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
