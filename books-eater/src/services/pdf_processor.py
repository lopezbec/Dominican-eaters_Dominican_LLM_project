"""PDF to markdown converter using pdfplumber."""

import logging
from pathlib import Path
from typing import Optional
import pdfplumber

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Processor for converting PDF files to markdown text.
    """
    
    @staticmethod
    def pdf_to_markdown(pdf_path: str) -> Optional[str]:
        """
        Convert a PDF file to markdown text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Markdown text or None if conversion fails
        """
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            markdown_text = []
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing PDF with {total_pages} pages: {pdf_path}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        
                        if text and text.strip():
                            markdown_text.append(f"## Page {page_num}\n\n{text.strip()}\n")
                        
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        continue
            
            if markdown_text:
                result = "\n".join(markdown_text)
                logger.info(f"Successfully converted PDF: {len(result)} characters")
                return result
            else:
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return ""
            
        except Exception as e:
            logger.error(f"Error converting PDF to markdown: {e}")
            return None
    
    @staticmethod
    def extract_text_simple(pdf_path: str) -> Optional[str]:
        """
        Extract plain text from PDF without markdown formatting.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Plain text or None if extraction fails
        """
        try:
            text_parts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    try:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text.strip())
                    except Exception:
                        continue
            
            return "\n\n".join(text_parts) if text_parts else ""
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None
    
    @staticmethod
    def get_pdf_info(pdf_path: str) -> dict:
        """
        Get basic information about a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return {
                    'pages': len(pdf.pages),
                    'metadata': pdf.metadata
                }
        except Exception as e:
            logger.error(f"Error getting PDF info: {e}")
            return {'pages': 0, 'metadata': {}}
