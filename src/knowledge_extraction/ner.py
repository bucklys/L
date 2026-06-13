"""
Named Entity Recognition Module - Extract entities from text
"""
from typing import List, Dict, Tuple
import logging
import spacy
from transformers import pipeline

logger = logging.getLogger(__name__)


class EntityRecognizer:
    """Extract named entities from text using spaCy and Hugging Face models"""
    
    def __init__(self, model_name: str = "en_core_web_sm", use_transformers: bool = False):
        """
        Initialize Entity Recognizer
        
        Args:
            model_name: spaCy model name (e.g., "en_core_web_sm", "zh_core_web_sm")
            use_transformers: Whether to use Transformers-based NER
        """
        self.model_name = model_name
        self.use_transformers = use_transformers
        
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Model {model_name} not found. Install it with: python -m spacy download {model_name}")
            self.nlp = None
        
        if use_transformers:
            try:
                self.hf_ner = pipeline("ner", model="dbmdz/bert-base-multilingual-cased")
                logger.info("Loaded Transformers NER model")
            except Exception as e:
                logger.error(f"Failed to load Transformers NER model: {e}")
                self.hf_ner = None
        else:
            self.hf_ner = None
    
    def extract_entities_spacy(self, text: str) -> List[Dict]:
        """
        Extract entities using spaCy
        
        Args:
            text: Input text
        
        Returns:
            List of entities with text, label, start, end
        """
        if not self.nlp:
            logger.warning("spaCy model not loaded")
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'source': 'spacy'
            })
        
        return entities
    
    def extract_entities_transformers(self, text: str, max_length: int = 512) -> List[Dict]:
        """
        Extract entities using Transformers
        
        Args:
            text: Input text
            max_length: Maximum text length for model
        
        Returns:
            List of entities
        """
        if not self.hf_ner:
            logger.warning("Transformers NER model not loaded")
            return []
        
        # Truncate text if needed
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            results = self.hf_ner(text)
            entities = []
            
            for result in results:
                entities.append({
                    'text': result['word'],
                    'label': result['entity'],
                    'score': result['score'],
                    'source': 'transformers'
                })
            
            return entities
        except Exception as e:
            logger.error(f"Error in Transformers NER: {e}")
            return []
    
    def extract_entities(self, text: str, combine: bool = True) -> List[Dict]:
        """
        Extract entities from text
        
        Args:
            text: Input text
            combine: Whether to combine results from multiple models
        
        Returns:
            List of extracted entities
        """
        entities = self.extract_entities_spacy(text)
        
        if self.use_transformers and self.hf_ner:
            hf_entities = self.extract_entities_transformers(text)
            if combine:
                entities.extend(hf_entities)
            else:
                entities = hf_entities
        
        return entities
    
    def get_entity_types(self, text: str) -> Dict[str, List[str]]:
        """
        Get entities grouped by type
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with entity types as keys and list of entity texts as values
        """
        entities = self.extract_entities(text)
        entity_types = {}
        
        for entity in entities:
            label = entity['label']
            text_val = entity['text']
            
            if label not in entity_types:
                entity_types[label] = []
            
            if text_val not in entity_types[label]:
                entity_types[label].append(text_val)
        
        return entity_types


def extract_entities(text: str, model_name: str = "en_core_web_sm") -> List[Dict]:
    """
    Convenience function to extract entities
    
    Args:
        text: Input text
        model_name: spaCy model name
    
    Returns:
        List of entities
    """
    recognizer = EntityRecognizer(model_name=model_name)
    return recognizer.extract_entities(text)
