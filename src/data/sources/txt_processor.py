import os
from pathlib import Path

from src.schemas import DocumentProcessor

class TXTProcessor(DocumentProcessor):
    """Process TXT documents"""
    
    def extract_text(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                raise Exception(f"Error processing TXT {file_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing TXT {file_path}: {str(e)}")
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        try:
            return {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_type': 'txt',
                'file_size': os.path.getsize(file_path),
                'created': os.path.getctime(file_path),
                'modified': os.path.getmtime(file_path)
            }
        except Exception as e:
            return {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_type': 'txt',
                'error': str(e)
            }
