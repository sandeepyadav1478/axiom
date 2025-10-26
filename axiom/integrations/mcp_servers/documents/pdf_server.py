"""PDF Processing MCP Server Implementation.

Provides PDF document processing through MCP protocol:
- Text extraction from PDFs
- Table extraction (financial tables)
- 10-K/10-Q SEC filing parsing
- OCR for scanned documents
- Keyword search
- Financial metrics extraction
- Document summarization
- Document comparison
"""

import io
import logging
import re
from pathlib import Path
from typing import Any, Optional

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    PdfReader = None

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    pytesseract = None
    convert_from_path = None

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    tabula = None

logger = logging.getLogger(__name__)


class PDFProcessingMCPServer:
    """PDF Processing MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ocr_enabled = config.get("ocr_enabled", True)
        self.ocr_language = config.get("ocr_language", "eng")
        self.extract_tables = config.get("extract_tables", True)
        self.extract_images = config.get("extract_images", False)
        
        # Check dependencies
        if not PDFPLUMBER_AVAILABLE and not PYPDF2_AVAILABLE:
            logger.warning("No PDF libraries available. Install pdfplumber or PyPDF2")

    async def extract_text(
        self,
        pdf_path: str,
        pages: Optional[list[int]] = None,
        method: str = "pdfplumber",
    ) -> dict[str, Any]:
        """Extract text from PDF.

        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to extract (None for all)
            method: Extraction method (pdfplumber, pypdf2)

        Returns:
            Extracted text
        """
        try:
            path = Path(pdf_path)
            if not path.exists():
                return {
                    "success": False,
                    "error": f"PDF file not found: {pdf_path}",
                }
            
            text_by_page = []
            full_text = ""
            
            if method == "pdfplumber" and PDFPLUMBER_AVAILABLE:
                with pdfplumber.open(path) as pdf:
                    pages_to_process = pages or list(range(len(pdf.pages)))
                    
                    for page_num in pages_to_process:
                        if page_num < len(pdf.pages):
                            page = pdf.pages[page_num]
                            text = page.extract_text() or ""
                            text_by_page.append({
                                "page": page_num + 1,
                                "text": text,
                            })
                            full_text += text + "\n\n"
            
            elif method == "pypdf2" and PYPDF2_AVAILABLE:
                with open(path, 'rb') as file:
                    pdf = PdfReader(file)
                    pages_to_process = pages or list(range(len(pdf.pages)))
                    
                    for page_num in pages_to_process:
                        if page_num < len(pdf.pages):
                            page = pdf.pages[page_num]
                            text = page.extract_text() or ""
                            text_by_page.append({
                                "page": page_num + 1,
                                "text": text,
                            })
                            full_text += text + "\n\n"
            else:
                return {
                    "success": False,
                    "error": f"Extraction method '{method}' not available or not supported",
                }
            
            return {
                "success": True,
                "pdf_path": str(path),
                "method": method,
                "pages": text_by_page,
                "full_text": full_text.strip(),
                "page_count": len(text_by_page),
                "total_chars": len(full_text),
            }

        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to extract text: {str(e)}",
                "pdf_path": pdf_path,
            }

    async def extract_tables(
        self,
        pdf_path: str,
        pages: Optional[list[int]] = None,
        method: str = "pdfplumber",
    ) -> dict[str, Any]:
        """Extract tables from PDF.

        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to extract (None for all)
            method: Extraction method (pdfplumber, tabula)

        Returns:
            Extracted tables
        """
        try:
            path = Path(pdf_path)
            if not path.exists():
                return {
                    "success": False,
                    "error": f"PDF file not found: {pdf_path}",
                }
            
            tables_by_page = []
            
            if method == "pdfplumber" and PDFPLUMBER_AVAILABLE:
                with pdfplumber.open(path) as pdf:
                    pages_to_process = pages or list(range(len(pdf.pages)))
                    
                    for page_num in pages_to_process:
                        if page_num < len(pdf.pages):
                            page = pdf.pages[page_num]
                            tables = page.extract_tables()
                            
                            if tables:
                                for idx, table in enumerate(tables):
                                    tables_by_page.append({
                                        "page": page_num + 1,
                                        "table_index": idx,
                                        "rows": len(table),
                                        "cols": len(table[0]) if table else 0,
                                        "data": table,
                                    })
            
            elif method == "tabula" and TABULA_AVAILABLE:
                pages_str = ",".join(map(str, pages)) if pages else "all"
                dfs = tabula.read_pdf(str(path), pages=pages_str, multiple_tables=True)
                
                for idx, df in enumerate(dfs):
                    tables_by_page.append({
                        "page": "unknown",
                        "table_index": idx,
                        "rows": len(df),
                        "cols": len(df.columns),
                        "data": df.values.tolist(),
                        "columns": df.columns.tolist(),
                    })
            else:
                return {
                    "success": False,
                    "error": f"Extraction method '{method}' not available or not supported",
                }
            
            return {
                "success": True,
                "pdf_path": str(path),
                "method": method,
                "tables": tables_by_page,
                "table_count": len(tables_by_page),
            }

        except Exception as e:
            logger.error(f"Failed to extract tables from {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to extract tables: {str(e)}",
                "pdf_path": pdf_path,
            }

    async def extract_10k_sections(
        self,
        pdf_path: str,
    ) -> dict[str, Any]:
        """Extract sections from 10-K SEC filing.

        Args:
            pdf_path: Path to 10-K PDF file

        Returns:
            Extracted sections
        """
        try:
            # Extract full text first
            result = await self.extract_text(pdf_path)
            if not result["success"]:
                return result
            
            text = result["full_text"]
            
            # Define 10-K section patterns
            sections = {
                "business": r"Item\s+1[.\s]+Business",
                "risk_factors": r"Item\s+1A[.\s]+Risk\s+Factors",
                "properties": r"Item\s+2[.\s]+Properties",
                "legal_proceedings": r"Item\s+3[.\s]+Legal\s+Proceedings",
                "mine_safety": r"Item\s+4[.\s]+Mine\s+Safety",
                "market_for_stock": r"Item\s+5[.\s]+Market\s+for",
                "selected_financial_data": r"Item\s+6[.\s]+Selected\s+Financial",
                "md_and_a": r"Item\s+7[.\s]+Management's\s+Discussion",
                "financial_statements": r"Item\s+8[.\s]+Financial\s+Statements",
                "changes_in_accounting": r"Item\s+9[.\s]+Changes\s+in",
                "controls": r"Item\s+9A[.\s]+Controls\s+and\s+Procedures",
            }
            
            extracted_sections = {}
            
            for section_name, pattern in sections.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = match.start()
                    # Find next section or end
                    next_match = None
                    for next_pattern in sections.values():
                        if next_pattern != pattern:
                            next_match = re.search(next_pattern, text[start+100:], re.IGNORECASE)
                            if next_match:
                                break
                    
                    if next_match:
                        end = start + 100 + next_match.start()
                        section_text = text[start:end]
                    else:
                        section_text = text[start:start+5000]  # Max 5000 chars
                    
                    extracted_sections[section_name] = {
                        "found": True,
                        "text": section_text.strip(),
                        "length": len(section_text),
                    }
                else:
                    extracted_sections[section_name] = {
                        "found": False,
                        "text": "",
                        "length": 0,
                    }
            
            return {
                "success": True,
                "pdf_path": pdf_path,
                "sections": extracted_sections,
                "sections_found": sum(1 for s in extracted_sections.values() if s["found"]),
            }

        except Exception as e:
            logger.error(f"Failed to extract 10-K sections from {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to extract 10-K sections: {str(e)}",
                "pdf_path": pdf_path,
            }

    async def extract_10q_data(
        self,
        pdf_path: str,
    ) -> dict[str, Any]:
        """Extract data from 10-Q SEC filing.

        Args:
            pdf_path: Path to 10-Q PDF file

        Returns:
            Extracted data
        """
        try:
            # Extract full text
            result = await self.extract_text(pdf_path)
            if not result["success"]:
                return result
            
            text = result["full_text"]
            
            # Extract key 10-Q sections
            sections = {
                "financial_statements": r"Part\s+I.*?Item\s+1[.\s]+Financial\s+Statements",
                "md_and_a": r"Item\s+2[.\s]+Management's\s+Discussion",
                "quantitative_qualitative": r"Item\s+3[.\s]+Quantitative\s+and\s+Qualitative",
                "controls": r"Item\s+4[.\s]+Controls\s+and\s+Procedures",
            }
            
            extracted_sections = {}
            
            for section_name, pattern in sections.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = match.start()
                    section_text = text[start:start+3000]  # Max 3000 chars
                    extracted_sections[section_name] = {
                        "found": True,
                        "text": section_text.strip(),
                    }
                else:
                    extracted_sections[section_name] = {
                        "found": False,
                        "text": "",
                    }
            
            # Extract tables for financial data
            tables_result = await self.extract_tables(pdf_path)
            
            return {
                "success": True,
                "pdf_path": pdf_path,
                "sections": extracted_sections,
                "tables": tables_result.get("tables", []) if tables_result["success"] else [],
                "sections_found": sum(1 for s in extracted_sections.values() if s["found"]),
            }

        except Exception as e:
            logger.error(f"Failed to extract 10-Q data from {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to extract 10-Q data: {str(e)}",
                "pdf_path": pdf_path,
            }

    async def ocr_scan(
        self,
        pdf_path: str,
        pages: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Perform OCR on scanned PDF.

        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to OCR (None for all)

        Returns:
            OCR text
        """
        if not OCR_AVAILABLE:
            return {
                "success": False,
                "error": "OCR not available. Install pytesseract and pdf2image",
            }
        
        try:
            path = Path(pdf_path)
            if not path.exists():
                return {
                    "success": False,
                    "error": f"PDF file not found: {pdf_path}",
                }
            
            # Convert PDF to images
            if pages:
                images = convert_from_path(path, first_page=min(pages)+1, last_page=max(pages)+1)
            else:
                images = convert_from_path(path)
            
            ocr_results = []
            full_text = ""
            
            for idx, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang=self.ocr_language)
                ocr_results.append({
                    "page": idx + 1,
                    "text": text,
                })
                full_text += text + "\n\n"
            
            return {
                "success": True,
                "pdf_path": str(path),
                "language": self.ocr_language,
                "pages": ocr_results,
                "full_text": full_text.strip(),
                "page_count": len(ocr_results),
            }

        except Exception as e:
            logger.error(f"Failed to OCR scan {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to perform OCR: {str(e)}",
                "pdf_path": pdf_path,
            }

    async def find_keywords(
        self,
        pdf_path: str,
        keywords: list[str],
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """Search for keywords in PDF.

        Args:
            pdf_path: Path to PDF file
            keywords: List of keywords to search
            case_sensitive: Case-sensitive search

        Returns:
            Keyword occurrences
        """
        try:
            # Extract text
            result = await self.extract_text(pdf_path)
            if not result["success"]:
                return result
            
            text = result["full_text"]
            if not case_sensitive:
                text = text.lower()
            
            keyword_results = {}
            
            for keyword in keywords:
                search_keyword = keyword if case_sensitive else keyword.lower()
                matches = []
                
                # Find all occurrences
                start = 0
                while True:
                    pos = text.find(search_keyword, start)
                    if pos == -1:
                        break
                    
                    # Extract context (50 chars before and after)
                    context_start = max(0, pos - 50)
                    context_end = min(len(text), pos + len(search_keyword) + 50)
                    context = text[context_start:context_end]
                    
                    matches.append({
                        "position": pos,
                        "context": context,
                    })
                    
                    start = pos + 1
                
                keyword_results[keyword] = {
                    "count": len(matches),
                    "matches": matches[:10],  # Limit to first 10
                }
            
            return {
                "success": True,
                "pdf_path": pdf_path,
                "keywords": keyword_results,
                "total_matches": sum(r["count"] for r in keyword_results.values()),
            }

        except Exception as e:
            logger.error(f"Failed to find keywords in {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to find keywords: {str(e)}",
                "pdf_path": pdf_path,
            }

    async def extract_metrics(
        self,
        pdf_path: str,
        metric_patterns: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Extract financial metrics from PDF.

        Args:
            pdf_path: Path to PDF file
            metric_patterns: Custom metric patterns (regex)

        Returns:
            Extracted metrics
        """
        try:
            # Extract text
            result = await self.extract_text(pdf_path)
            if not result["success"]:
                return result
            
            text = result["full_text"]
            
            # Default metric patterns
            default_patterns = {
                "revenue": r"(?:revenue|sales|net\s+sales)[:|\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|M|B)?",
                "net_income": r"(?:net\s+income|net\s+profit)[:|\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|M|B)?",
                "eps": r"(?:earnings\s+per\s+share|EPS)[:|\s]+\$?\s*([\d,]+\.?\d*)",
                "ebitda": r"EBITDA[:|\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|M|B)?",
                "total_assets": r"(?:total\s+assets)[:|\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|M|B)?",
                "total_debt": r"(?:total\s+debt)[:|\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|M|B)?",
            }
            
            patterns = metric_patterns or default_patterns
            
            extracted_metrics = {}
            
            for metric_name, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Take first match
                    value = matches[0].replace(',', '')
                    try:
                        extracted_metrics[metric_name] = {
                            "value": float(value),
                            "raw": matches[0],
                            "found": True,
                        }
                    except ValueError:
                        extracted_metrics[metric_name] = {
                            "value": None,
                            "raw": matches[0],
                            "found": True,
                        }
                else:
                    extracted_metrics[metric_name] = {
                        "value": None,
                        "raw": None,
                        "found": False,
                    }
            
            return {
                "success": True,
                "pdf_path": pdf_path,
                "metrics": extracted_metrics,
                "metrics_found": sum(1 for m in extracted_metrics.values() if m["found"]),
            }

        except Exception as e:
            logger.error(f"Failed to extract metrics from {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to extract metrics: {str(e)}",
                "pdf_path": pdf_path,
            }

    async def summarize_document(
        self,
        pdf_path: str,
        max_length: int = 500,
    ) -> dict[str, Any]:
        """Generate summary of PDF document.

        Args:
            pdf_path: Path to PDF file
            max_length: Maximum summary length

        Returns:
            Document summary
        """
        try:
            # Extract text
            result = await self.extract_text(pdf_path)
            if not result["success"]:
                return result
            
            text = result["full_text"]
            
            # Simple extractive summarization (first paragraphs + keywords)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Take first few paragraphs that fit in max_length
            summary = ""
            for para in paragraphs[:5]:
                if len(summary) + len(para) < max_length:
                    summary += para + "\n\n"
                else:
                    break
            
            # Extract key statistics
            metrics_result = await self.extract_metrics(pdf_path)
            key_metrics = {
                k: v for k, v in metrics_result.get("metrics", {}).items()
                if v["found"]
            }
            
            return {
                "success": True,
                "pdf_path": pdf_path,
                "summary": summary.strip(),
                "summary_length": len(summary),
                "key_metrics": key_metrics,
                "total_length": len(text),
                "page_count": result["page_count"],
            }

        except Exception as e:
            logger.error(f"Failed to summarize {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to summarize document: {str(e)}",
                "pdf_path": pdf_path,
            }

    async def compare_documents(
        self,
        pdf_path1: str,
        pdf_path2: str,
    ) -> dict[str, Any]:
        """Compare two PDF documents.

        Args:
            pdf_path1: Path to first PDF
            pdf_path2: Path to second PDF

        Returns:
            Comparison results
        """
        try:
            # Extract text from both
            result1 = await self.extract_text(pdf_path1)
            result2 = await self.extract_text(pdf_path2)
            
            if not result1["success"] or not result2["success"]:
                return {
                    "success": False,
                    "error": "Failed to extract text from one or both documents",
                }
            
            text1 = result1["full_text"]
            text2 = result2["full_text"]
            
            # Basic comparison metrics
            len1 = len(text1)
            len2 = len(text2)
            
            # Word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            similarity = overlap / union if union > 0 else 0
            
            # Extract metrics from both
            metrics1 = await self.extract_metrics(pdf_path1)
            metrics2 = await self.extract_metrics(pdf_path2)
            
            metric_changes = {}
            for metric_name in metrics1.get("metrics", {}).keys():
                m1 = metrics1["metrics"][metric_name]
                m2 = metrics2["metrics"].get(metric_name, {})
                
                if m1.get("found") and m2.get("found"):
                    v1 = m1.get("value")
                    v2 = m2.get("value")
                    if v1 and v2:
                        change = ((v2 - v1) / v1) * 100 if v1 != 0 else 0
                        metric_changes[metric_name] = {
                            "value1": v1,
                            "value2": v2,
                            "change_percent": change,
                        }
            
            return {
                "success": True,
                "document1": {
                    "path": pdf_path1,
                    "length": len1,
                    "page_count": result1["page_count"],
                },
                "document2": {
                    "path": pdf_path2,
                    "length": len2,
                    "page_count": result2["page_count"],
                },
                "similarity": similarity,
                "metric_changes": metric_changes,
            }

        except Exception as e:
            logger.error(f"Failed to compare documents: {e}")
            return {
                "success": False,
                "error": f"Failed to compare documents: {str(e)}",
            }


def get_server_definition() -> dict[str, Any]:
    """Get PDF Processing MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "pdf",
        "category": "documents",
        "description": "PDF document processing (text extraction, tables, financial data, OCR)",
        "tools": [
            {
                "name": "extract_text",
                "description": "Extract text from PDF document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to PDF file",
                        },
                        "pages": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Specific pages to extract (0-indexed, omit for all)",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pdfplumber", "pypdf2"],
                            "description": "Extraction method",
                            "default": "pdfplumber",
                        },
                    },
                    "required": ["pdf_path"],
                },
            },
            {
                "name": "extract_tables",
                "description": "Extract tables from PDF (for financial statements)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to PDF file",
                        },
                        "pages": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Specific pages to extract (0-indexed)",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pdfplumber", "tabula"],
                            "description": "Extraction method",
                            "default": "pdfplumber",
                        },
                    },
                    "required": ["pdf_path"],
                },
            },
            {
                "name": "extract_10k_sections",
                "description": "Parse 10-K SEC filing sections",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to 10-K PDF file",
                        }
                    },
                    "required": ["pdf_path"],
                },
            },
            {
                "name": "extract_10q_data",
                "description": "Parse 10-Q SEC filing data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to 10-Q PDF file",
                        }
                    },
                    "required": ["pdf_path"],
                },
            },
            {
                "name": "ocr_scan",
                "description": "Perform OCR on scanned PDF",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to PDF file",
                        },
                        "pages": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Specific pages to OCR (0-indexed)",
                        },
                    },
                    "required": ["pdf_path"],
                },
            },
            {
                "name": "find_keywords",
                "description": "Search for keywords in PDF",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to PDF file",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords to search for",
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Case-sensitive search",
                            "default": False,
                        },
                    },
                    "required": ["pdf_path", "keywords"],
                },
            },
            {
                "name": "extract_metrics",
                "description": "Extract financial metrics from PDF",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to PDF file",
                        },
                        "metric_patterns": {
                            "type": "object",
                            "description": "Custom metric regex patterns (optional)",
                        },
                    },
                    "required": ["pdf_path"],
                },
            },
            {
                "name": "summarize_document",
                "description": "Generate AI-powered summary of PDF",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to PDF file",
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum summary length",
                            "default": 500,
                        },
                    },
                    "required": ["pdf_path"],
                },
            },
            {
                "name": "compare_documents",
                "description": "Compare two PDF documents (e.g., consecutive filings)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path1": {
                            "type": "string",
                            "description": "Path to first PDF",
                        },
                        "pdf_path2": {
                            "type": "string",
                            "description": "Path to second PDF",
                        },
                    },
                    "required": ["pdf_path1", "pdf_path2"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "high",
            "category": "documents",
            "requires": [
                "pdfplumber>=0.10.3",
                "PyPDF2>=3.0.1",
                "pytesseract>=0.3.10",
                "tabula-py>=2.8.2",
                "pdf2image>=1.16.3",
            ],
            "performance_target": "<2s for text, <5s for tables",
        },
    }