"""
Relation Extraction Module - Extract relationships between entities
"""
from typing import List, Dict, Tuple
import logging
import re
from transformers import pipeline

logger = logging.getLogger(__name__)


class RelationExtractor:
    """Extract relationships between entities"""
    
    def __init__(self, use_transformers: bool = True):
        """
        Initialize Relation Extractor
        
        Args:
            use_transformers: Whether to use Transformers-based models
        """
        self.use_transformers = use_transformers
        
        if use_transformers:
            try:
                # Zero-shot relation extraction
                self.zero_shot_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                logger.info("Loaded Transformers relation extraction model")
            except Exception as e:
                logger.error(f"Failed to load Transformers model: {e}")
                self.zero_shot_classifier = None
        else:
            self.zero_shot_classifier = None
    
    def extract_relations_pattern(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        Extract relations using pattern matching
        
        Args:
            text: Input text
            entities: List of extracted entities
        
        Returns:
            List of relations (subject, predicate, object)
        """
        relations = []
        
        # Common relation patterns
        patterns = [
            r'(\w+)\s+(?:is|are)\s+(?:a|an)\s+(\w+)',  # X is a Y
            r'(\w+)\s+(?:was|were)\s+(?:a|an)\s+(\w+)',  # X was a Y
            r'(\w+)\s+founded\s+(?:by|in)\s+(\w+)',  # X founded by Y / in Z
            r'(\w+)\s+(?:located|situated)\s+(?:in|at)\s+(\w+)',  # X located in Y
            r'(\w+)\s+(?:works|works for)\s+(\w+)',  # X works for Y
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1)
                obj = match.group(2)
                # Determine relation type from pattern
                relation_type = pattern.split('\\s+')[1] if len(pattern.split('\\s+')) > 1 else "unknown"
                
                relations.append({
                    'subject': subject,
                    'relation': relation_type,
                    'object': obj,
                    'confidence': 0.5,
                    'source': 'pattern'
                })
        
        return relations
    
    def extract_relations_transformers(
        self,
        text: str,
        entities: List[Dict],
        candidate_relations: List[str] = None
    ) -> List[Dict]:
        """
        Extract relations using zero-shot classification
        
        Args:
            text: Input text
            entities: List of extracted entities
            candidate_relations: Possible relation types to classify
        
        Returns:
            List of relations with confidence scores
        """
        if not self.zero_shot_classifier or not candidate_relations:
            return []
        
        relations = []
        
        # Generate entity pairs
        entity_pairs = []
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                entity_pairs.append((ent1, ent2))
        
        # Classify relations for each pair
        for ent1, ent2 in entity_pairs:
            try:
                premise = f"{ent1['text']} and {ent2['text']}"
                result = self.zero_shot_classifier(premise, candidate_relations)
                
                if result['scores'][0] > 0.5:  # Confidence threshold
                    relations.append({
                        'subject': ent1['text'],
                        'subject_type': ent1['label'],
                        'relation': result['labels'][0],
                        'object': ent2['text'],
                        'object_type': ent2['label'],
                        'confidence': result['scores'][0],
                        'source': 'transformers'
                    })
            except Exception as e:
                logger.debug(f"Error classifying relation: {e}")
        
        return relations
    
    def extract_relations(
        self,
        text: str,
        entities: List[Dict],
        candidate_relations: List[str] = None,
        use_pattern: bool = True
    ) -> List[Dict]:
        """
        Extract relations from text
        
        Args:
            text: Input text
            entities: List of extracted entities
            candidate_relations: Possible relation types
            use_pattern: Whether to use pattern-based extraction
        
        Returns:
            List of extracted relations
        """
        relations = []
        
        if use_pattern:
            relations.extend(self.extract_relations_pattern(text, entities))
        
        if self.use_transformers and self.zero_shot_classifier:
            if not candidate_relations:
                candidate_relations = [
                    "is a",
                    "works for",
                    "located in",
                    "created by",
                    "similar to",
                    "related to"
                ]
            relations.extend(self.extract_relations_transformers(text, entities, candidate_relations))
        
        return relations
    
    def deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """
        Remove duplicate relations
        
        Args:
            relations: List of relations
        
        Returns:
            Deduplicated list of relations
        """
        seen = set()
        deduplicated = []
        
        for rel in relations:
            key = (rel['subject'], rel['relation'], rel['object'])
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)
        
        return deduplicated


def extract_relations(
    text: str,
    entities: List[Dict],
    candidate_relations: List[str] = None
) -> List[Dict]:
    """
    Convenience function to extract relations
    
    Args:
        text: Input text
        entities: List of entities
        candidate_relations: Possible relation types
    
    Returns:
        List of relations
    """
    extractor = RelationExtractor()
    return extractor.extract_relations(text, entities, candidate_relations)
