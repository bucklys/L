"""
Summarizer Module - Generate summaries from text
"""
from typing import List, Optional
import logging
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Summarizer:
    """Generate text summaries"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize Summarizer
        
        Args:
            model_name: Transformer model name for summarization
        """
        self.model_name = model_name
        
        try:
            self.summarizer = pipeline("summarization", model=model_name)
            logger.info(f"Loaded summarization model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            self.summarizer = None
    
    def summarize_transformers(
        self,
        text: str,
        min_length: int = 50,
        max_length: int = 150,
        num_beams: int = 4
    ) -> str:
        """
        Summarize using Transformers model
        
        Args:
            text: Input text
            min_length: Minimum summary length
            max_length: Maximum summary length
            num_beams: Number of beams for beam search
        
        Returns:
            Summary text
        """
        if not self.summarizer:
            logger.warning("Summarizer not loaded")
            return ""
        
        # Truncate long texts
        max_input_length = 1024
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        try:
            summary = self.summarizer(
                text,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return ""
    
    def summarize_extractive(
        self,
        text: str,
        num_sentences: int = 3
    ) -> str:
        """
        Extract key sentences as summary
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
        
        Returns:
            Summary text
        """
        try:
            sentences = sent_tokenize(text)
            
            # Simple scoring based on word frequency
            words = nltk.word_tokenize(text.lower())
            from nltk.corpus import stopwords
            
            try:
                stop_words = set(stopwords.words('english'))
            except:
                # Fallback if stopwords not available
                stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
            
            word_freq = {}
            for word in words:
                if word.isalnum() and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score sentences
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                for word in nltk.word_tokenize(sentence.lower()):
                    if word in word_freq:
                        sentence_scores[i] = sentence_scores.get(i, 0) + word_freq[word]
            
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
            top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Restore original order
            
            summary = " ".join([sentences[i] for i, _ in top_sentences])
            return summary
        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            return ""
    
    def summarize(
        self,
        text: str,
        method: str = "transformers",
        **kwargs
    ) -> str:
        """
        Generate summary from text
        
        Args:
            text: Input text
            method: "transformers" or "extractive"
            **kwargs: Additional arguments for specific methods
        
        Returns:
            Summary text
        """
        if method == "transformers":
            return self.summarize_transformers(text, **kwargs)
        elif method == "extractive":
            return self.summarize_extractive(text, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def summarize_bullets(self, text: str, num_bullets: int = 5) -> List[str]:
        """
        Generate bullet point summary
        
        Args:
            text: Input text
            num_bullets: Number of bullet points
        
        Returns:
            List of bullet point summaries
        """
        try:
            sentences = sent_tokenize(text)
            # Use extractive method to get key sentences
            summary = self.summarize_extractive(text, num_sentences=num_bullets)
            bullets = sent_tokenize(summary)
            return bullets
        except Exception as e:
            logger.error(f"Error in bullet point summarization: {e}")
            return []


def summarize(text: str, method: str = "transformers", **kwargs) -> str:
    """
    Convenience function to summarize text
    
    Args:
        text: Input text
        method: "transformers" or "extractive"
        **kwargs: Additional arguments
    
    Returns:
        Summary text
    """
    summarizer = Summarizer()
    return summarizer.summarize(text, method=method, **kwargs)
