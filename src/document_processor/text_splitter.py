"""
Text Splitter Module - Split documents into chunks
"""
from typing import List
import logging

logger = logging.getLogger(__name__)


class TextSplitter:
    """Split text into chunks with overlap"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize TextSplitter
        
        Args:
            chunk_size: Size of each chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def split_by_size(self, text: str) -> List[str]:
        """
        Split text by fixed size with overlap
        
        Args:
            text: Input text
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Find a good break point (space or newline) near the chunk boundary
            if end < len(text):
                # Look back for a space or newline
                break_point = text.rfind(' ', start + self.chunk_size - 100, end)
                if break_point == -1 or break_point < start:
                    break_point = text.rfind('\n', start + self.chunk_size - 100, end)
                if break_point == -1 or break_point < start:
                    break_point = end
                end = break_point
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap
        
        return chunks
    
    def split_documents(self, documents: List[str], strategy: str = "size") -> List[str]:
        """
        Split multiple documents into chunks
        
        Args:
            documents: List of documents (text strings)
            strategy: "size", "sentence", or "paragraph"
        
        Returns:
            List of chunks
        """
        all_chunks = []
        
        for doc in documents:
            if strategy == "size":
                chunks = self.split_by_size(doc)
            elif strategy == "sentence":
                chunks = self.split_by_sentences(doc)
            elif strategy == "paragraph":
                chunks = self.split_by_paragraphs(doc)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            all_chunks.extend(chunks)
        
        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Convenience function to split text
    
    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    splitter = TextSplitter(chunk_size=chunk_size, overlap=overlap)
    return splitter.split_by_size(text)
