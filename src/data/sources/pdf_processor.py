import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from io import BytesIO
import logging

# PDF processing libraries
import PyPDF as PyPDF2
import pymupdf as fitz  # PyMuPDF - better for layout and images
import pdfplumber  # Excellent for tables
import camelot  # Advanced table extraction

# Image processing
from PIL import Image

# OCR
import pytesseract

# Table processing
import pandas as pd
import tabula

from src.schemas import DocumentProcessor, ExtractedFigure, ExtractedImage, ExtractedTable

class PDFProcessor(DocumentProcessor):
    """PDF processor with table, figure, and image extraction capabilities"""
    
    def __init__(self, 
                 extract_images: bool = True,
                 extract_tables: bool = True,
                 extract_figures: bool = True,
                 perform_ocr: bool = True,
                 min_image_size: Tuple[int, int] = (50, 50),
                 table_extraction_methods: List[str] = None):
        """
        Initialize the enhanced PDF processor
        
        Args:
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            extract_figures: Whether to extract figures/charts
            perform_ocr: Whether to perform OCR on images
            min_image_size: Minimum size (width, height) for image extraction
            table_extraction_methods: Methods to use for table extraction
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures
        self.perform_ocr = perform_ocr
        self.min_image_size = min_image_size
        
        if table_extraction_methods is None:
            self.table_extraction_methods = ['pdfplumber', 'camelot', 'tabula']
        else:
            self.table_extraction_methods = table_extraction_methods
        
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, file_path: str) -> str:
        """Extract text with better layout preservation"""
        try:
            text_parts = []
            
            # Use PyMuPDF for better text extraction with layout
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    # Extract text blocks with position info
                    blocks = page.get_text("blocks")
                    page_text = []
                    
                    # Sort blocks by vertical position, then horizontal
                    blocks.sort(key=lambda b: (b[1], b[0]))
                    
                    for block in blocks:
                        if len(block) >= 5 and block[4].strip():
                            page_text.append(block[4].strip())
                    
                    if page_text:
                        text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                        text_parts.append("\n\n".join(page_text))
            
            return "\n".join(text_parts).strip()
            
        except Exception as e:
            # Fallback to PyPDF2
            self.logger.warning(f"PyMuPDF failed, falling back to PyPDF2: {str(e)}")
            return self._extract_text_pypdf2(file_path)
    
    def _extract_text_pypdf2(self, file_path: str) -> str:
        """Fallback text extraction using PyPDF2"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {str(e)}")
    
    def extract_images(self, file_path: str) -> List[ExtractedImage]:
        """Extract images from PDF"""
        if not self.extract_images:
            return []
        
        images = []
        
        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    # Get image list from page
                    image_list = page.get_images(full=True)
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Extract image data
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            # Create PIL Image for processing
                            pil_image = Image.open(BytesIO(image_bytes))
                            width, height = pil_image.size
                            
                            # Skip small images (likely decorative)
                            if width < self.min_image_size[0] or height < self.min_image_size[1]:
                                continue
                            
                            # Get image position on page
                            image_rects = page.get_image_rects(xref)
                            bbox = image_rects[0] if image_rects else (0, 0, width, height)
                            
                            # Generate unique ID
                            image_id = self._generate_image_id(file_path, page_num, img_index)
                            
                            # Perform OCR if enabled
                            extracted_text = None
                            if self.perform_ocr:
                                try:
                                    extracted_text = pytesseract.image_to_string(pil_image).strip()
                                except Exception as ocr_e:
                                    self.logger.warning(f"OCR failed for image {image_id}: {str(ocr_e)}")
                            
                            # Create extracted image object
                            extracted_image = ExtractedImage(
                                image_id=image_id,
                                page_number=page_num + 1,
                                bbox=bbox,
                                image_data=image_bytes,
                                image_format=image_ext,
                                width=width,
                                height=height,
                                file_size=len(image_bytes),
                                extracted_text=extracted_text
                            )
                            
                            images.append(extracted_image)
                            
                        except Exception as img_e:
                            self.logger.warning(f"Failed to extract image {img_index} from page {page_num}: {str(img_e)}")
                            continue
            
        except Exception as e:
            self.logger.error(f"Error extracting images from {file_path}: {str(e)}")
        
        return images
    
    def extract_tables(self, file_path: str) -> List[ExtractedTable]:
        """Extract tables using multiple methods for best results"""
        if not self.extract_tables:
            return []
        
        tables = []
        
        # Method 1: PDFPlumber (good for simple tables)
        if 'pdfplumber' in self.table_extraction_methods:
            tables.extend(self._extract_tables_pdfplumber(file_path))
        
        # Method 2: Camelot (good for complex tables)
        if 'camelot' in self.table_extraction_methods:
            tables.extend(self._extract_tables_camelot(file_path))
        
        # Method 3: Tabula (Java-based, good for complex layouts)
        if 'tabula' in self.table_extraction_methods:
            tables.extend(self._extract_tables_tabula(file_path))
        
        # Remove duplicates and rank by confidence
        tables = self._deduplicate_tables(tables)
        
        return tables
    
    def _extract_tables_pdfplumber(self, file_path: str) -> List[ExtractedTable]:
        """Extract tables using pdfplumber"""
        tables = []
        
        try:
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_index, table_data in enumerate(page_tables):
                        if not table_data or len(table_data) < 2:
                            continue
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        
                        # Clean DataFrame
                        df = df.dropna(how='all').fillna('')
                        
                        if df.empty:
                            continue
                        
                        table_id = self._generate_table_id(file_path, page_num, table_index, 'pdfplumber')
                        
                        extracted_table = ExtractedTable(
                            table_id=table_id,
                            page_number=page_num + 1,
                            bbox=None,  # pdfplumber doesn't provide bbox easily
                            dataframe=df,
                            confidence=0.8,  # pdfplumber is generally reliable
                            extraction_method='pdfplumber',
                            csv_data=df.to_csv(index=False),
                            html_data=df.to_html(index=False)
                        )
                        
                        tables.append(extracted_table)
                        
        except Exception as e:
            self.logger.warning(f"PDFPlumber table extraction failed: {str(e)}")
        
        return tables
    
    def _extract_tables_camelot(self, file_path: str) -> List[ExtractedTable]:
        """Extract tables using Camelot"""
        tables = []
        
        try:
            # Extract tables using both lattice and stream methods
            for flavor in ['lattice', 'stream']:
                try:
                    camelot_tables = camelot.read_pdf(file_path, flavor=flavor, pages='all')
                    
                    for table_index, table in enumerate(camelot_tables):
                        df = table.df
                        
                        # Clean DataFrame
                        df = df.dropna(how='all').fillna('')
                        
                        if df.empty or df.shape[0] < 2:
                            continue
                        
                        table_id = self._generate_table_id(file_path, table.page, table_index, f'camelot_{flavor}')
                        
                        extracted_table = ExtractedTable(
                            table_id=table_id,
                            page_number=table.page,
                            bbox=tuple(table._bbox) if hasattr(table, '_bbox') else None,
                            dataframe=df,
                            confidence=table.accuracy / 100.0 if hasattr(table, 'accuracy') else 0.7,
                            extraction_method=f'camelot_{flavor}',
                            csv_data=df.to_csv(index=False),
                            html_data=df.to_html(index=False)
                        )
                        
                        tables.append(extracted_table)
                        
                except Exception as flavor_e:
                    self.logger.debug(f"Camelot {flavor} extraction failed: {str(flavor_e)}")
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Camelot table extraction failed: {str(e)}")
        
        return tables
    
    def _extract_tables_tabula(self, file_path: str) -> List[ExtractedTable]:
        """Extract tables using Tabula"""
        tables = []
        
        try:
            # Read all tables from all pages
            tabula_tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            
            for table_index, df in enumerate(tabula_tables):
                # Clean DataFrame
                df = df.dropna(how='all').fillna('')
                
                if df.empty or df.shape[0] < 2:
                    continue
                
                table_id = self._generate_table_id(file_path, 0, table_index, 'tabula')
                
                extracted_table = ExtractedTable(
                    table_id=table_id,
                    page_number=1,  # Tabula doesn't easily provide page numbers
                    bbox=None,
                    dataframe=df,
                    confidence=0.6,  # Tabula can be less reliable
                    extraction_method='tabula',
                    csv_data=df.to_csv(index=False),
                    html_data=df.to_html(index=False)
                )
                
                tables.append(extracted_table)
                
        except Exception as e:
            self.logger.warning(f"Tabula table extraction failed: {str(e)}")
        
        return tables
    
    def extract_figures(self, file_path: str) -> List[ExtractedFigure]:
        """Extract figures and charts from PDF"""
        if not self.extract_figures:
            return []
        
        figures = []
        
        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    # Extract drawings/graphics (potential figures)
                    drawings = page.get_drawings()
                    
                    for drawing_index, drawing in enumerate(drawings):
                        try:
                            # Get drawing bounds
                            bbox = drawing.get('rect', (0, 0, 100, 100))
                            
                            # Create image from drawing area
                            clip = fitz.Rect(bbox)
                            mat = fitz.Matrix(2, 2)  # 2x scaling for better quality
                            pix = page.get_pixmap(matrix=mat, clip=clip)
                            
                            if pix.width < self.min_image_size[0] or pix.height < self.min_image_size[1]:
                                continue
                            
                            image_bytes = pix.tobytes("png")
                            
                            # Perform OCR to extract any text in the figure
                            extracted_text = None
                            if self.perform_ocr:
                                try:
                                    pil_image = Image.open(BytesIO(image_bytes))
                                    extracted_text = pytesseract.image_to_string(pil_image).strip()
                                except Exception:
                                    pass
                            
                            figure_id = self._generate_figure_id(file_path, page_num, drawing_index)
                            
                            figure = ExtractedFigure(
                                figure_id=figure_id,
                                page_number=page_num + 1,
                                bbox=bbox,
                                figure_type=self._classify_figure_type(drawing),
                                image_data=image_bytes,
                                extracted_text=extracted_text
                            )
                            
                            figures.append(figure)
                            
                        except Exception as fig_e:
                            self.logger.warning(f"Failed to extract figure {drawing_index} from page {page_num}: {str(fig_e)}")
                            continue
            
        except Exception as e:
            self.logger.error(f"Error extracting figures from {file_path}: {str(e)}")
        
        return figures
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract enhanced metadata including content statistics"""
        try:
            metadata = {}
            
            # Basic file info
            metadata.update({
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_type': 'pdf',
                'file_size': os.path.getsize(file_path)
            })
            
            # PDF-specific metadata
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                pdf_metadata = reader.metadata or {}
                
                metadata.update({
                    'num_pages': len(reader.pages),
                    'title': pdf_metadata.get('/Title', ''),
                    'author': pdf_metadata.get('/Author', ''),
                    'subject': pdf_metadata.get('/Subject', ''),
                    'creator': pdf_metadata.get('/Creator', ''),
                    'creation_date': str(pdf_metadata.get('/CreationDate', '')),
                    'modification_date': str(pdf_metadata.get('/ModDate', ''))
                })
            
            # Content statistics
            if self.extract_images or self.extract_tables or self.extract_figures:
                content_stats = {}
                
                if self.extract_images:
                    images = self.extract_images(file_path)
                    content_stats['num_images'] = len(images)
                    content_stats['total_image_size'] = sum(img.file_size for img in images)
                
                if self.extract_tables:
                    tables = self.extract_tables(file_path)
                    content_stats['num_tables'] = len(tables)
                    content_stats['total_table_cells'] = sum(table.dataframe.size for table in tables)
                
                if self.extract_figures:
                    figures = self.extract_figures(file_path)
                    content_stats['num_figures'] = len(figures)
                
                metadata['content_statistics'] = content_stats
            
            return metadata
            
        except Exception as e:
            return {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_type': 'pdf',
                'error': str(e)
            }
    
    def extract_all(self, file_path: str) -> Dict[str, Any]:
        """Extract all content from PDF: text, images, tables, and figures"""
        result = {
            'text': self.extract_text(file_path),
            'metadata': self.extract_metadata(file_path),
            'images': [],
            'tables': [],
            'figures': []
        }
        
        if self.extract_images:
            result['images'] = self.extract_images(file_path)
        
        if self.extract_tables:
            result['tables'] = self.extract_tables(file_path)
        
        if self.extract_figures:
            result['figures'] = self.extract_figures(file_path)
        
        return result
    
    # Helper methods
    def _generate_image_id(self, file_path: str, page_num: int, img_index: int) -> str:
        """Generate unique image ID"""
        base_string = f"{file_path}_{page_num}_{img_index}_image"
        return hashlib.md5(base_string.encode()).hexdigest()[:16]
    
    def _generate_table_id(self, file_path: str, page_num: int, table_index: int, method: str) -> str:
        """Generate unique table ID"""
        base_string = f"{file_path}_{page_num}_{table_index}_{method}_table"
        return hashlib.md5(base_string.encode()).hexdigest()[:16]
    
    def _generate_figure_id(self, file_path: str, page_num: int, fig_index: int) -> str:
        """Generate unique figure ID"""
        base_string = f"{file_path}_{page_num}_{fig_index}_figure"
        return hashlib.md5(base_string.encode()).hexdigest()[:16]
    
    def _classify_figure_type(self, drawing: Dict) -> str:
        """Classify figure type based on drawing properties"""
        # Simple classification logic - can be enhanced
        if 'items' in drawing:
            items = drawing['items']
            if any('curve' in str(item).lower() for item in items):
                return 'chart'
            elif any('rect' in str(item).lower() for item in items):
                return 'diagram'
        return 'figure'
    
    def _deduplicate_tables(self, tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """Remove duplicate tables and keep the one with highest confidence"""
        if not tables:
            return []
        
        # Group by page and approximate size
        grouped = {}
        for table in tables:
            key = (table.page_number, table.dataframe.shape)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(table)
        
        # Keep the table with highest confidence from each group
        deduplicated = []
        for group in grouped.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Sort by confidence and take the best one
                best_table = max(group, key=lambda t: t.confidence)
                deduplicated.append(best_table)
        
        return deduplicated
    
    def save_extracted_content(self, extracted_content: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Save extracted content to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        base_name = Path(extracted_content['metadata']['file_name']).stem
        
        # Save text
        text_path = output_path / f"{base_name}_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(extracted_content['text'])
        file_paths['text'] = str(text_path)
        
        # Save images
        if extracted_content['images']:
            images_dir = output_path / "images"
            images_dir.mkdir(exist_ok=True)
            
            for img in extracted_content['images']:
                img_path = images_dir / f"{base_name}_{img.image_id}.{img.image_format}"
                with open(img_path, 'wb') as f:
                    f.write(img.image_data)
                
                # Save OCR text if available
                if img.extracted_text:
                    ocr_path = images_dir / f"{base_name}_{img.image_id}_ocr.txt"
                    with open(ocr_path, 'w', encoding='utf-8') as f:
                        f.write(img.extracted_text)
        
        # Save tables
        if extracted_content['tables']:
            tables_dir = output_path / "tables"
            tables_dir.mkdir(exist_ok=True)
            
            for table in extracted_content['tables']:
                csv_path = tables_dir / f"{base_name}_{table.table_id}.csv"
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write(table.csv_data)
                
                html_path = tables_dir / f"{base_name}_{table.table_id}.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(table.html_data)
        
        # Save figures
        if extracted_content['figures']:
            figures_dir = output_path / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            for fig in extracted_content['figures']:
                fig_path = figures_dir / f"{base_name}_{fig.figure_id}.png"
                with open(fig_path, 'wb') as f:
                    f.write(fig.image_data)
                
                # Save extracted text if available
                if fig.extracted_text:
                    text_path = figures_dir / f"{base_name}_{fig.figure_id}_text.txt"
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(fig.extracted_text)
        