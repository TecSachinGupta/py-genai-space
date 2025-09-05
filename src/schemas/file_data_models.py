from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pandas import DataFrame 

@dataclass
class ExtractedImage:
    """Container for extracted image data"""
    image_id: str
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    image_data: bytes
    image_format: str
    width: int
    height: int
    file_size: int
    extracted_text: Optional[str] = None  # OCR text if available
    description: Optional[str] = None

@dataclass
class ExtractedTable:
    """Container for extracted table data"""
    table_id: str
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]]
    dataframe: DataFrame
    confidence: float
    extraction_method: str
    csv_data: str
    html_data: str

@dataclass
class ExtractedFigure:
    """Container for extracted figure/chart data"""
    figure_id: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    figure_type: str  # 'chart', 'diagram', 'graph', etc.
    image_data: bytes
    extracted_text: Optional[str] = None
    caption: Optional[str] = None

@dataclass
class ExtractedContent:
    """Complete extracted content from DOCX document"""
    text: str
    images: List[ExtractedImage]
    tables: List[ExtractedTable]
    figures: List[ExtractedFigure]
    headers: List[str]
    footers: List[str]
    hyperlinks: List[Dict[str, str]]
    comments: List[Dict[str, str]]
    metadata: Dict[str, Any]