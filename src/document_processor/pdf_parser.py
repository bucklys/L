"""
PDF Parser Module - Extract text and images from PDF files
"""
import os
import io
from typing import List, Dict, Tuple, Optional
import PyPDF2
import pdfplumber
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse PDF files (both text-based and image-based)"""
    
    def __init__(self, pdf_path: str, extract_images: bool = True, use_ocr: bool = False):
        """
        Initialize PDF Parser
        
        Args:
            pdf_path: Path to PDF file
            extract_images: Whether to extract images from image-based PDFs
            use_ocr: Whether to use OCR for image-based content
        """
        self.pdf_path = pdf_path
        self.extract_images = extract_images
        self.use_ocr = use_ocr
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def extract_text_pypdf2(self) -> str:
        """Extract text using PyPDF2 (works for text-based PDFs)"""
        try:
            text = []
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text.append(f"--- Page {page_num + 1} ---\n{page_text}")
            return "\n".join(text)
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return ""
    
    def extract_text_pdfplumber(self) -> str:
        """Extract text using pdfplumber (better for complex layouts)"""
        try:
            text = []
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    
                    # Extract tables if any
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            table_text = "\n".join([" | ".join(str(cell) if cell else "" for cell in row) for row in table])
                            text.append(f"\n[TABLE]\n{table_text}\n[/TABLE]")
            
            return "\n".join(text)
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return ""
    
    def extract_images_from_pdf(self) -> List[Tuple[str, bytes]]:
        """Extract images from PDF"""
        images = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    for img_idx, img in enumerate(page.images):
                        # Extract image object
                        image_name = f"page_{page_num + 1}_img_{img_idx + 1}.png"
                        try:
                            # Save image bytes
                            images.append((image_name, img['stream'].get_rawdata()))
                        except Exception as e:
                            logger.warning(f"Could not extract image {image_name}: {e}")
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
        
        return images
    
    def ocr_image(self, image_data: bytes) -> str:
        """Extract text from image using OCR (requires pytesseract)"""
        try:
            import pytesseract
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image)
            return text
        except ImportError:
            logger.warning("pytesseract not installed. Install it for OCR support: pip install pytesseract")
            return ""
        except Exception as e:
            logger.error(f"Error in OCR: {e}")
            return ""
    
    def extract_text(self, method: str = "pdfplumber") -> str:
        """
        Extract text from PDF
        
        Args:
            method: "pypdf2" or "pdfplumber" (pdfplumber is more reliable)
        
        Returns:
            Extracted text from PDF
        """
        if method == "pdfplumber":
            return self.extract_text_pdfplumber()
        elif method == "pypdf2":
            return self.extract_text_pypdf2()
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def extract_all(self, use_ocr: bool = False) -> Dict:
        """
        Extract all content from PDF (text + images)
        
        Returns:
            Dictionary with 'text' and 'images' keys
        """
        result = {
            'text': self.extract_text(),
            'images': [],
            'metadata': {
                'file_path': self.pdf_path,
                'file_size': os.path.getsize(self.pdf_path)
            }
        }
        
        if self.extract_images:
            images = self.extract_images_from_pdf()
            result['images'] = images
            
            # Use OCR on images if requested
            if use_ocr or self.use_ocr:
                ocr_texts = []
                for img_name, img_data in images:
                    ocr_text = self.ocr_image(img_data)
                    if ocr_text.strip():
                        ocr_texts.append(f"[Image: {img_name}]\n{ocr_text}")
                
                if ocr_texts:
                    result['text'] += "\n\n" + "\n\n".join(ocr_texts)
        
        return result


def parse_pdf(pdf_path: str, extract_images: bool = True, use_ocr: bool = False) -> str:
    """
    Convenience function to parse PDF and return text
    
    Args:
        pdf_path: Path to PDF file
        extract_images: Whether to extract images
        use_ocr: Whether to use OCR
    
    Returns:
        Extracted text
    """
    parser = PDFParser(pdf_path, extract_images=extract_images, use_ocr=use_ocr)
    return parser.extract_text()
