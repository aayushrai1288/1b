import os
import json
import re
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, ValidationError, Field
import pdfplumber
from PyPDF2 import PdfReader
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Optional OCR Imports ---
# Set up robust error handling for OCR dependencies
OCR_AVAILABLE = False
try:
    from pytesseract import image_to_string
    from pdf2image import convert_from_path
    # Test if the poppler-utils (providing pdfinfo) are in the system's PATH
    import subprocess
    subprocess.run(['pdfinfo', '--version'], capture_output=True, check=True, text=True)
    OCR_AVAILABLE = True
    logging.info("OCR functionality is available (pytesseract and poppler found).")
except (ImportError, FileNotFoundError, subprocess.CalledProcessError):
    OCR_AVAILABLE = False
    logging.warning("OCR functionality is disabled: pytesseract or poppler-utils not found in system PATH.")

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Pydantic Schemas for Configuration ---

class DocumentConfig(BaseModel):
    filename: str
    title: str = Field(..., description="Human-readable title of the PDF document")

class InputConfig(BaseModel):
    challenge_info: Dict[str, Any]
    documents: List[DocumentConfig]
    persona: Dict[str, str]
    job_to_be_done: Dict[str, str]


def load_input_config(input_json_path: str) -> InputConfig:
    """Load and validate input configuration from a JSON file using Pydantic."""
    if not os.path.isfile(input_json_path):
        logging.error(f"Input config not found: {input_json_path}")
        raise FileNotFoundError(f"Input config not found: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    try:
        return InputConfig(**data)
    except ValidationError as e:
        logging.error(f"Invalid input configuration schema: {e}")
        raise


# --- Text Processing and Extraction Functions ---

def clean_text_for_embedding(text: str) -> str:
    """Enhanced text cleaning for better embedding quality."""
    # 1. Normalize whitespace (spaces, newlines, tabs) to a single space
    txt = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Remove common boilerplate text like page numbers and copyright notices
    txt = re.sub(r'Page\s+\d+\s+of\s+\d+', '', txt, flags=re.IGNORECASE)
    txt = re.sub(r'^\d+\s*$', '', txt, flags=re.MULTILINE)  # Remove standalone page numbers
    txt = re.sub(r'©.*?\d{4}', '', txt)  # Remove copyright notices
    
    # 3. Expand common abbreviations
    glossary = {
        'HR': 'Human Resources', 'PDF': 'Portable Document Format',
        'CEO': 'Chief Executive Officer', 'AI': 'Artificial Intelligence',
        'ML': 'Machine Learning'
    }
    for k, v in glossary.items():
        txt = re.sub(rf'\b{k}\b', v, txt)
    
    return txt


def extract_tables_and_lists(page) -> List[Dict[str, Any]]:
    """Extract tables and structured lists as separate content blocks from a page."""
    content_blocks = []
    
    # Extract tables using pdfplumber's built-in functionality
    tables = page.find_tables()
    for i, table in enumerate(tables):
        try:
            table_data = table.extract()
            # Ensure table has at least a header and one row of data
            if table_data and len(table_data) > 1:
                table_text = "\n".join([" | ".join(map(str, row)) for row in table_data])
                content_blocks.append({
                    'type': 'table',
                    'content': table_text,
                    'position': f"table_{i}",
                    'bbox': table.bbox
                })
        except Exception as e:
            logging.warning(f"Could not extract table {i} on page {page.page_number}: {e}")
            
    # Extract lists using regex patterns
    text = page.extract_text() or ""
    list_patterns = [
        r'((?:^[\s]*[•·▪▫○◦‣⁃*+-]\s+.+\n?)+)',  # Bullet/symbol lists
        r'((?:^[\s]*\d+[\.\)]\s+.+\n?)+)',      # Numbered lists
        r'((?:^[\s]*[a-zA-Z][\.\)]\s+.+\n?)+)' # Lettered lists
    ]
    
    for pattern in list_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for i, match in enumerate(matches):
            if len(match.strip().split('\n')) >= 2: # Must have at least 2 items
                content_blocks.append({
                    'type': 'list',
                    'content': match.strip(),
                    'position': f"list_{i}_{pattern[:10]}",
                    'bbox': None
                })
    return content_blocks


def extract_figure_captions(text: str) -> List[str]:
    """Extract figure and table captions that might contain important context."""
    caption_pattern = r'(?:Figure|Fig\.?|Table|Chart)\s+\d+[:\.]?\s*([^\n.]{20,150})'
    return re.findall(caption_pattern, text, re.IGNORECASE)


def extract_pages_text_enhanced(pdf_path: str) -> List[Dict[str, Any]]:
    """Enhanced text extraction with OCR fallback and structured content detection."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""

                # OCR fallback if text is sparse and OCR is available
                if len(raw_text) < 100 and OCR_AVAILABLE:
                    try:
                        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
                        if images:
                            ocr_text = image_to_string(images[0])
                            if len(ocr_text) > len(raw_text):
                                raw_text = ocr_text
                                logging.info(f"Used OCR for page {page_num} in {os.path.basename(pdf_path)}")
                    except Exception as e:
                        logging.warning(f"OCR failed for page {page_num} in {os.path.basename(pdf_path)}: {e}")
                
                if not raw_text.strip():
                    raw_text = f"[NO TEXT EXTRACTED - Page {page_num}]"

                structured_content = extract_tables_and_lists(page)
                captions = extract_figure_captions(raw_text)
                cleaned_text = clean_text_for_embedding(raw_text)
                
                pages_content.append({
                    'page_number': page_num,
                    'text': cleaned_text,
                    'structured_content': structured_content,
                    'captions': captions,
                })
    except Exception as e:
        logging.error(f"Fatal error processing PDF {pdf_path}: {e}")
        return [{'page_number': 1, 'text': f"[ERROR: Could not process PDF. Reason: {e}]", 'structured_content': [], 'captions': []}]
    
    return pages_content


def enhanced_section_detection(pages_data: List[Dict[str, Any]], pdf_filename: str) -> List[Dict[str, Any]]:
    """Detect sections based on headings and layout cues."""
    sections = []
    # Regex to find headings (e.g., ALL CAPS, Title Case, Numbered)
    heading_pattern = re.compile(
        r'^(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Za-z\s]{5,50}|[A-Z\s]{5,50})$'
    )

    for page in pages_data:
        text_by_line = page['text'].split('\n')
        current_heading = f"Page {page['page_number']} Content"
        current_content = []

        for line in text_by_line:
            match = heading_pattern.match(line.strip())
            if match:
                # If we found a new heading, save the previous section
                if current_content:
                    sections.append({
                        'document': pdf_filename,
                        'page_number': page['page_number'],
                        'section_title': current_heading,
                        'text': ' '.join(current_content).strip(),
                    })
                current_heading = match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section from the page
        if current_content:
            sections.append({
                'document': pdf_filename,
                'page_number': page['page_number'],
                'section_title': current_heading,
                'text': ' '.join(current_content).strip(),
            })
            
    return [s for s in sections if len(s['text']) > 100] # Filter out very short/empty sections


# --- Retrieval and Ranking Logic ---

def generate_query_variants(persona: str, task: str) -> List[str]:
    """Generate multiple query variants to improve retrieval coverage."""
    # This is a simplified, heuristic-based approach. For more complex cases,
    # a language model could be used to generate more diverse queries.
    combined_text = f"{persona} {task}"
    keywords = re.findall(r'\b[A-Za-z-]{4,}\b', combined_text) # Extract simple keywords
    
    variants = [
        task,
        f"{persona} needs to {task}",
        f"Information about '{task}' for a {persona}",
        f"Key points for {persona}: {task}",
        ' '.join(keywords)
    ]
    return list(set(variants))[:5] # Return unique variants, max 5


class WeakSupervisionClassifier:
    """A lightweight classifier to refine section relevance scores using pseudo-labels."""
    def __init__(self):
        """Correctly initializes the classifier."""
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.classifier = LogisticRegression(random_state=42, class_weight='balanced')
        self.is_trained = False
    
    def create_pseudo_labels(self, scores: List[float]) -> List[int]:
        """Create pseudo-labels based on cross-encoder scores."""
        if not scores:
            return []
        threshold = np.percentile(scores, 75)  # Label top 25% as relevant (1)
        return [1 if score > threshold else 0 for score in scores]
    
    def train(self, section_texts: List[str], labels: List[int]):
        """Train the logistic regression classifier."""
        if len(set(labels)) < 2:  # Requires at least two classes to train
            logging.warning("Skipping weak classifier training: only one class present in pseudo-labels.")
            return
        
        X = self.tfidf.fit_transform(section_texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def predict_proba(self, section_texts: List[str]) -> np.ndarray:
        """Predict relevance probabilities for sections."""
        if not self.is_trained:
            return np.full(len(section_texts), 0.5)  # Return neutral score if not trained
        
        X = self.tfidf.transform(section_texts)
        return self.classifier.predict_proba(X)[:, 1]


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item_idx in enumerate(ranking):
            scores[item_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: -item[1])


# --- Main Processing Function ---

def process_challenge_1b_enhanced(input_json_path: str, pdfs_directory: str, output_json_path: str):
    """Enhanced RAG pipeline for processing the challenge."""
    config = load_input_config(input_json_path)
    persona = config.persona['role']
    task = config.job_to_be_done['task']
    
    logging.info(f"Starting processing for task: '{task}'")

    # 1. Initialize models
    bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    weak_classifier = WeakSupervisionClassifier()

    # 2. Extract and sectionize all documents
    all_sections, processed_docs = [], []
    for doc in config.documents:
        pdf_path = os.path.join(pdfs_directory, doc.filename)
        if not os.path.exists(pdf_path):
            logging.warning(f"PDF file not found, skipping: {pdf_path}")
            continue
        
        pages = extract_pages_text_enhanced(pdf_path)
        sections = enhanced_section_detection(pages, doc.filename)
        all_sections.extend(sections)
        processed_docs.append(doc.filename)

    if not all_sections:
        logging.error("No sections were extracted from any documents. Aborting.")
        return

    section_texts = [s['text'] for s in all_sections]
    logging.info(f"Extracted {len(all_sections)} sections from {len(processed_docs)} documents.")

    # 3. Multi-Query Retrieval and Fusion
    query_variants = generate_query_variants(persona, task)
    section_embeddings = bi_encoder.encode(section_texts, convert_to_tensor=True, show_progress_bar=True)
    
    all_rankings = []
    for query in query_variants:
        query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
        similarities = cosine_similarity(query_embedding.reshape(1, -1), section_embeddings)[0]
        ranking = np.argsort(-similarities).tolist()
        all_rankings.append(ranking)

    fused_ranking = reciprocal_rank_fusion(all_rankings)
    top_candidate_indices = [idx for idx, _ in fused_ranking[:50]] # Get top 50 candidates for reranking

    # 4. Cross-Encoder Reranking and Weak Supervision
    primary_query = task
    rerank_pairs = [(primary_query, all_sections[i]['text']) for i in top_candidate_indices]
    ce_scores = cross_encoder.predict(rerank_pairs, show_progress_bar=True)
    
    candidate_texts = [all_sections[i]['text'] for i in top_candidate_indices]
    pseudo_labels = weak_classifier.create_pseudo_labels(ce_scores.tolist())
    weak_classifier.train(candidate_texts, pseudo_labels)
    weak_scores = weak_classifier.predict_proba(candidate_texts)
    
    # Combine scores (70% cross-encoder, 30% weak classifier)
    final_scores = 0.7 * ce_scores + 0.3 * weak_scores
    
    # 5. Final Ranking and Selection
    reranked_results = sorted(zip(top_candidate_indices, final_scores), key=lambda x: -x[1], reverse=True)
    top_5_indices = [idx for idx, score in reranked_results[:5]]

    # 6. Generate Output JSON
    extracted_sections = []
    for rank, idx in enumerate(top_5_indices, start=1):
        sec = all_sections[idx]
        extracted_sections.append({
            "document": sec['document'],
            "section_title": sec['section_title'],
            "importance_rank": rank,
            "page_number": sec['page_number']
        })

    subsection_analysis = []
    for idx in top_5_indices:
        sec = all_sections[idx]
        # For refined text, we find the 3 most relevant sentences
        sentences = re.split(r'(?<=[.!?])\s+', sec['text'])
        sentences = [s for s in sentences if len(s) > 15] # Filter short sentences
        
        if len(sentences) > 3:
            sent_embeddings = bi_encoder.encode(sentences)
            query_embedding = bi_encoder.encode(primary_query).reshape(1, -1)
            sent_scores = cosine_similarity(query_embedding, sent_embeddings)[0]
            top_sent_indices = np.argsort(-sent_scores)[:3]
            refined_text = " ".join([sentences[i] for i in sorted(top_sent_indices)])
        else:
            refined_text = sec['text']

        subsection_analysis.append({
            "document": sec['document'],
            "refined_text": refined_text,
            "page_number": sec['page_number']
        })

    output_data = {
        "metadata": {
            "input_documents": processed_docs,
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    logging.info(f"Successfully wrote output to {output_json_path}")


# Backward compatibility wrapper
def process_challenge_1b(input_json_path: str, pdfs_directory: str, output_json_path: str):
    """Wrapper to maintain backward compatibility."""
    return process_challenge_1b_enhanced(input_json_path, pdfs_directory, output_json_path)


# ========================== EXAMPLE USAGE ==========================
if __name__ == '__main__':
    # 1. Suppress warnings (optional)
    import warnings
    warnings.filterwarnings("ignore")

    # 2. Define the paths to YOUR input and output files
    # This should match the directory structure you created.
    base_dir = './Collection 1'
    pdfs_dir = os.path.join(base_dir, 'pdfs')
    input_json_path = os.path.join(base_dir, 'input.json')
    output_json_path = os.path.join(base_dir, 'output.json')

    # 3. Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # 4. Run the main processing function on your files
    try:
        logging.info(f"Starting processing using config: {input_json_path}")
        process_challenge_1b_enhanced(input_json_path, pdfs_dir, output_json_path)
        logging.info(f"Processing finished. Output saved to: {output_json_path}")
        
    except FileNotFoundError as e:
        logging.error(f"A required file or directory was not found: {e}")
        logging.error("Please ensure your 'input_config' directory is structured correctly with your JSON in 'config/' and your PDF(s) in 'pdfs/'.")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}")