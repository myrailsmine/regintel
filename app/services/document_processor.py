"""
Document processing service with multiple extraction strategies
"""

import os
import time
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
import logging

import fitz  # PyMuPDF
import pdfplumber
import spacy
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np

from app.core.config import settings, ProcessingConfig

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Rich document metadata"""
    doc_id: str
    filename: str
    file_path: str
    file_size: int
    page_count: int
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    language: str = "en"
    document_type: Optional[str] = None
    processing_timestamp: Optional[str] = None
    content_hash: Optional[str] = None
    extracted_entities: List[Dict] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    quality_score: float = 0.0

@dataclass
class ChunkMetadata:
    """Granular chunk metadata"""
    chunk_id: str
    doc_id: str
    page_number: int
    position: Tuple[float, float, float, float]  # x1, y1, x2, y2
    chunk_type: str  # text, table, image, header, footer
    sequence_number: int
    character_count: int
    entities: List[Dict] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)  # Related chunk IDs
    confidence_score: float = 0.0
    parent_section: Optional[str] = None

class BaseExtractor:
    """Abstract base class for PDF extractors"""
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Extract content from PDF"""
        raise NotImplementedError
    
    def extract_structured(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured content (tables, forms, etc.)"""
        raise NotImplementedError

class PyMuPDFExtractor(BaseExtractor):
    """PyMuPDF-based PDF extractor - fast, good for text"""
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        content = {"text_chunks": [], "metadata": [], "images": []}
        
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with positioning
                text_dict = page.get_text("dict")
                page_text = ""
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                page_text += span["text"] + " "
                        page_text += "\n"
                
                if page_text.strip():
                    content["text_chunks"].append(page_text.strip())
                    content["metadata"].append({
                        "doc_id": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "extraction_method": "pymupdf",
                        "char_count": len(page_text),
                        "bbox": page.rect
                    })
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    content["images"].append({
                        "page": page_num + 1,
                        "image_index": img_index,
                        "xref": img[0],
                        "bbox": page.get_image_bbox(img) if hasattr(page, 'get_image_bbox') else None
                    })
            
            doc.close()
            return content
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return {"text_chunks": [], "metadata": [], "images": [], "error": str(e)}
    
    def extract_structured(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables and forms using PyMuPDF"""
        structured_content = {"tables": [], "forms": []}
        
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Find tables
                try:
                    tables = page.find_tables()
                    for table_idx, table in enumerate(tables):
                        structured_content["tables"].append({
                            "page": page_num + 1,
                            "table_id": f"page_{page_num+1}_table_{table_idx+1}",
                            "bbox": table.bbox,
                            "data": table.extract()
                        })
                except:
                    pass  # Table extraction not available in all PyMuPDF versions
                
                # Extract form fields
                try:
                    widgets = page.widgets()
                    for widget in widgets:
                        structured_content["forms"].append({
                            "page": page_num + 1,
                            "field_name": widget.field_name,
                            "field_type": widget.field_type_string,
                            "field_value": widget.field_value,
                            "bbox": widget.rect
                        })
                except:
                    pass
            
            doc.close()
            return structured_content
            
        except Exception as e:
            logger.error(f"Structured extraction failed for {pdf_path}: {e}")
            return {"tables": [], "forms": [], "error": str(e)}

class PDFPlumberExtractor(BaseExtractor):
    """PDFPlumber-based extractor - better for tables and layout"""
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        content = {"text_chunks": [], "metadata": []}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        content["text_chunks"].append(text.strip())
                        content["metadata"].append({
                            "doc_id": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "extraction_method": "pdfplumber",
                            "char_count": len(text),
                            "page_width": page.width,
                            "page_height": page.height
                        })
            
            return content
            
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed for {pdf_path}: {e}")
            return {"text_chunks": [], "metadata": [], "error": str(e)}
    
    def extract_structured(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables using PDFPlumber"""
        structured_content = {"tables": []}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables or []):
                        if table:
                            structured_content["tables"].append({
                                "page": page_num + 1,
                                "table_id": f"page_{page_num+1}_table_{table_idx+1}",
                                "data": table,
                                "rows": len(table),
                                "columns": len(table[0]) if table else 0
                            })
            
            return structured_content
            
        except Exception as e:
            logger.error(f"Structured extraction failed for {pdf_path}: {e}")
            return {"tables": [], "error": str(e)}

class HybridExtractor(BaseExtractor):
    """Combines multiple extraction methods for best results"""
    
    def __init__(self):
        self.pymupdf_extractor = PyMuPDFExtractor()
        self.pdfplumber_extractor = PDFPlumberExtractor()
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        pymupdf_result = self.pymupdf_extractor.extract(pdf_path)
        pdfplumber_result = self.pdfplumber_extractor.extract(pdf_path)
        
        # Combine results, preferring better extraction
        combined_content = {
            "text_chunks": [],
            "metadata": [],
            "images": pymupdf_result.get("images", [])
        }
        
        # Choose better extraction per page
        pymupdf_chunks = pymupdf_result.get("text_chunks", [])
        pdfplumber_chunks = pdfplumber_result.get("text_chunks", [])
        pymupdf_meta = pymupdf_result.get("metadata", [])
        pdfplumber_meta = pdfplumber_result.get("metadata", [])
        
        max_pages = max(len(pymupdf_chunks), len(pdfplumber_chunks))
        
        for i in range(max_pages):
            pymupdf_text = pymupdf_chunks[i] if i < len(pymupdf_chunks) else ""
            pdfplumber_text = pdfplumber_chunks[i] if i < len(pdfplumber_chunks) else ""
            
            # Choose extraction with more content
            if len(pdfplumber_text) > len(pymupdf_text):
                if pdfplumber_text:
                    combined_content["text_chunks"].append(pdfplumber_text)
                    combined_content["metadata"].append(pdfplumber_meta[i])
            else:
                if pymupdf_text:
                    combined_content["text_chunks"].append(pymupdf_text)
                    combined_content["metadata"].append(pymupdf_meta[i])
        
        return combined_content
    
    def extract_structured(self, pdf_path: str) -> Dict[str, Any]:
        pymupdf_structured = self.pymupdf_extractor.extract_structured(pdf_path)
        pdfplumber_structured = self.pdfplumber_extractor.extract_structured(pdf_path)
