import os
import re
import logging
import requests

from pathlib import Path

from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

logger = logging.getLogger()

TERMINAL_WIDTH = os.get_terminal_size().columns

class PDFFileDownloader:
    """Downloads PDF files from the specified ArXiv URLs"""
    
    def __init__(self, download_dir='./pdfs'):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.urls = []
        self.downloaded_files = []

    def read_urls(self, file_path = './urls.txt'):
        """Read URLs from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.urls = [line.strip() for line in file if line.strip() and not line.startswith('#')]
            logger.info(f"Loaded {len(self.urls)} URLs from {file_path}")
        except FileNotFoundError:
            logger.exception(f"File not found: {file_path}")
            self.urls = []
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            self.urls = []

    def download_files(self):
        """Download all PDF files from the URLs"""
        self.downloaded_files = []
        
        for i, url in enumerate(self.urls, 1):
            logger.info(f"Downloading {i}/{len(self.urls)}: {url}")
            
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Generate filename based on ArXiv ID
                arxiv_id = url.split('/')[-1].replace('.pdf', '')
                temp_filename = f"arxiv_{arxiv_id}.pdf"
                temp_path = self.download_dir / temp_filename
                
                # Download the file
                with open(temp_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                
                # Verify it's a PDF
                if self._is_valid_pdf(temp_path):
                    self.downloaded_files.append({
                        'url': url,
                        'temp_path': temp_path,
                        'original_name': temp_filename,
                        'success': True
                    })
                    logger.info(f"Downloaded successfully: {temp_filename}")
                else:
                    temp_path.unlink()  # Delete invalid file
                    logger.info(f"Not a valid PDF file")
                    
            except Exception as e:
                logger.info(f"Failed to download: {e}")

    def smart_rename(self):
        """Rename files based on extracted titles"""
        for file_info in self.downloaded_files:
            if not file_info['success']:
                continue
                
            try:
                # Extract title from PDF
                title = self._extract_title(file_info['temp_path'])
                
                if title:
                    # Clean title for filename
                    clean_title = self._clean_filename(title)
                    new_filename = f"{clean_title}.pdf"
                else:
                    # Fallback to URL-based name
                    new_filename = self._generate_filename_from_url(file_info['url'])
                
                # Handle duplicates
                new_path = self._get_unique_path(new_filename)
                
                # Rename file
                file_info['temp_path'].rename(new_path)
                file_info['final_path'] = new_path
                file_info['final_name'] = new_path.name
                
                logger.info(f"Renamed: {file_info['original_name']} -> {new_path.name}")
                
            except Exception as e:
                logger.exception(f"Error renaming {file_info['temp_path']}: {e}")
                # Keep original temp name
                file_info['final_path'] = file_info['temp_path']
                file_info['final_name'] = file_info['temp_path'].name

    def execute(self, url_file_path = './urls.txt'):
        """Execute the complete download and rename process"""
        logger.info("Starting PDF download and smart rename process...")
        logger.info("=" * (TERMINAL_WIDTH - 50))
        
        # Step 1: Read URLs
        self.read_urls(url_file_path)
        if not self.urls:
            logger.info("No URLs to process. Exiting.")
            return
            
        # Step 2: Download files
        logger.info("Downloading files...")
        self.download_files()
        
        # Step 3: Smart rename
        logger.info("Renaming files based on content...")
        self.smart_rename()
        
        # Step 4: Summary
        self._print_summary()
        
    def _extract_title(self, pdf_path):
        """Extract title from PDF using pdfminer.six"""
        try:
            # First try to get title from PDF metadata
            with open(pdf_path, 'rb') as file:
                parser = PDFParser(file)
                document = PDFDocument(parser)
                
                if document.info and len(document.info) > 0:
                    info = document.info[0]
                    if 'Title' in info:
                        title = info['Title']
                        if title and isinstance(title, bytes):
                            title = title.decode('utf-8', errors='ignore')
                        elif title:
                            title = str(title)
                        
                        if title and title.strip():
                            return title.strip()
            
            # If no metadata title, extract from first page content
            text = extract_text(pdf_path, maxpages=1)
            if text:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # Look for title-like patterns (usually first few lines)
                for line in lines[:5]:  # Check first 5 lines
                    line = line.strip()
                    # Skip very short lines or lines that look like headers/footers
                    if len(line) > 10 and len(line) < 100:
                        # Avoid lines that are all caps (might be headers)
                        if not line.isupper():
                            return line
                
                # Fallback to first substantial line
                if lines:
                    return lines[0]
                    
        except Exception as e:
            logger.exception(f"Error extracting title: {e}")
            
        return None
        
    def _clean_filename(self, title):
        """Clean title to make it suitable for filename"""
        # Remove or replace invalid filename characters
        title = re.sub(r'[<>:"/\\|?*]', '', title)
        title = re.sub(r'\s+', ' ', title)  # Multiple spaces to single space
        title = title.strip()
        
        # Truncate if too long
        if len(title) > 100:
            title = title[:100].rsplit(' ', 1)[0]  # Cut at word boundary
            
        return title if title else "untitled"
        
    def _generate_filename_from_url(self, url):
        """Generate filename from URL as fallback"""
        try:
            filename = os.path.basename(url.split('?')[0])  # Remove query params
            if filename.endswith('.pdf'):
                return filename
            return f"{filename}.pdf" if filename else "downloaded.pdf"
        except:
            return "downloaded.pdf"
            
    def _get_unique_path(self, filename):
        """Get unique file path, handling duplicates"""
        path = self.download_dir / filename
        counter = 1
        
        while path.exists():
            name_part, ext = os.path.splitext(filename)
            new_filename = f"{name_part}_{counter}{ext}"
            path = self.download_dir / new_filename
            counter += 1
            
        return path
        
    def _is_valid_pdf(self, file_path):
        """Check if file is a valid PDF"""
        try:
            with open(file_path, 'rb') as file:
                header = file.read(5)
                return header.startswith(b'%PDF-')
        except:
            return False
            
    def _print_summary(self):
        """Print download summary"""
        logger.info("=" * (TERMINAL_WIDTH - 50))
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * (TERMINAL_WIDTH - 50))
        
        successful = sum(1 for f in self.downloaded_files if f['success'])
        total = len(self.urls)
        
        logger.info(f"Total URLs processed: {total}")
        logger.info(f"Successful downloads: {successful}")
        logger.info(f"Failed downloads: {total - successful}")
        
        if successful > 0:
            logger.info(f"Downloaded files:")
            for file_info in self.downloaded_files:
                if file_info['success']:
                    final_name = file_info.get('final_name', file_info['original_name'])
                    logger.info(f"  • {final_name}")