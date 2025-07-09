"""
Content & Domain Analysis for RLHF System

This module implements comprehensive content analysis including:
- Response quality assessment and scoring
- Domain-specific analysis and categorization
- Content complexity and readability metrics
- Topic modeling and clustering
- Quality correlation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import re
from collections import Counter, defaultdict
import json


@dataclass
class QualityMetrics:
    """Container for response quality metrics"""
    overall_score: float  # 0-1 overall quality
    readability_score: float  # Text readability
    complexity_score: float  # Content complexity
    coherence_score: float  # Logical coherence
    completeness_score: float  # Answer completeness
    accuracy_indicators: Dict[str, float]  # Various accuracy signals
    
    
@dataclass
class DomainAnalysis:
    """Domain-specific analysis results"""
    domain_name: str
    sample_count: int
    average_quality: float
    quality_variance: float
    common_topics: List[str]
    difficulty_level: str  # 'easy', 'medium', 'hard'
    performance_trends: Dict[str, float]
    

@dataclass
class ContentReport:
    """Comprehensive content analysis report"""
    quality_metrics: List[QualityMetrics]
    domain_analyses: List[DomainAnalysis]
    content_patterns: Dict[str, Any]
    improvement_suggestions: List[str]
    timestamp: datetime


class ContentAnalyzer:
    """
    Advanced content and domain analysis system.
    """
    
    def __init__(self):
        """Initialize the content analyzer."""
        self.domain_patterns = self._load_domain_patterns()
        self.quality_indicators = self._load_quality_indicators()
        
    def analyze_content(self, data: pd.DataFrame) -> ContentReport:
        """
        Perform comprehensive content analysis.
        
        Args:
            data: DataFrame with content data
            
        Returns:
            ContentReport with analysis results
        """
        # Quality analysis
        quality_metrics = self._analyze_quality(data)
        
        # Domain analysis
        domain_analyses = self._analyze_domains(data)
        
        # Content pattern detection
        content_patterns = self._detect_content_patterns(data)
        
        # Generate improvement suggestions
        suggestions = self._generate_suggestions(quality_metrics, domain_analyses)
        
        return ContentReport(
            quality_metrics=quality_metrics,
            domain_analyses=domain_analyses,
            content_patterns=content_patterns,
            improvement_suggestions=suggestions,
            timestamp=datetime.now()
        )
    
    def _analyze_quality(self, data: pd.DataFrame) -> List[QualityMetrics]:
        """Analyze response quality metrics."""
        quality_metrics = []
        
        for idx, row in data.iterrows():
            if 'response_a' in row and 'response_b' in row:
                # Analyze both responses
                for resp_col in ['response_a', 'response_b']:
                    if pd.notna(row[resp_col]):
                        metrics = self._calculate_quality_metrics(row[resp_col])
                        quality_metrics.append(metrics)
        
        return quality_metrics
    
    def _calculate_quality_metrics(self, text: str) -> QualityMetrics:
        """Calculate quality metrics for a single text."""
        # Basic quality indicators
        readability = self._calculate_readability(text)
        complexity = self._calculate_complexity(text)
        coherence = self._calculate_coherence(text)
        completeness = self._calculate_completeness(text)
        accuracy_indicators = self._get_accuracy_indicators(text)
        
        # Overall score (weighted average)
        overall = (readability * 0.2 + complexity * 0.2 + 
                  coherence * 0.3 + completeness * 0.3)
        
        return QualityMetrics(
            overall_score=overall,
            readability_score=readability,
            complexity_score=complexity,
            coherence_score=coherence,
            completeness_score=completeness,
            accuracy_indicators=accuracy_indicators
        )
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate text readability score."""
        if not text or len(text) < 10:
            return 0.0
        
        # Simple readability metrics
        sentences = len([s for s in text.split('.') if s.strip()])
        words = len(text.split())
        avg_sentence_length = words / max(1, sentences)
        
        # Penalize very long or very short sentences
        if avg_sentence_length > 25:
            readability = 0.6
        elif avg_sentence_length < 5:
            readability = 0.7
        else:
            readability = 0.9
        
        # Bonus for good structure
        if any(marker in text for marker in ['1.', '2.', '-', '•']):
            readability += 0.1
        
        return min(1.0, readability)
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate content complexity score."""
        if not text:
            return 0.0
        
        # Complexity indicators
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', text))  # Acronyms
        long_words = len([w for w in text.split() if len(w) > 8])
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        complexity_ratio = (technical_terms + long_words) / total_words
        
        # Scale to 0-1 range
        complexity_score = min(1.0, complexity_ratio * 3)
        
        return complexity_score
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate logical coherence score."""
        if not text or len(text) < 20:
            return 0.5
        
        # Simple coherence indicators
        coherence_score = 0.7  # Base score
        
        # Bonus for logical connectors
        connectors = ['therefore', 'however', 'furthermore', 'because', 'since']
        if any(conn in text.lower() for conn in connectors):
            coherence_score += 0.2
        
        # Bonus for structure
        if re.search(r'\d+\.|\-|\•', text):
            coherence_score += 0.1
        
        return min(1.0, coherence_score)
    
    def _calculate_completeness(self, text: str) -> float:
        """Calculate answer completeness score."""
        if not text:
            return 0.0
        
        # Length-based completeness
        word_count = len(text.split())
        
        if word_count < 10:
            completeness = 0.3
        elif word_count < 50:
            completeness = 0.7
        elif word_count < 200:
            completeness = 0.9
        else:
            completeness = 0.8  # Very long might be verbose
        
        # Bonus for comprehensive coverage
        if any(phrase in text.lower() for phrase in ['in summary', 'to conclude', 'examples include']):
            completeness += 0.1
        
        return min(1.0, completeness)
    
    def _get_accuracy_indicators(self, text: str) -> Dict[str, float]:
        """Get various accuracy indicators."""
        indicators = {}
        
        # Confidence language
        uncertain_phrases = ['might', 'could', 'possibly', 'perhaps', 'may']
        confident_phrases = ['definitely', 'certainly', 'clearly', 'obviously']
        
        uncertain_count = sum(1 for phrase in uncertain_phrases if phrase in text.lower())
        confident_count = sum(1 for phrase in confident_phrases if phrase in text.lower())
        
        indicators['confidence_level'] = confident_count / max(1, confident_count + uncertain_count)
        
        # Factual markers
        fact_markers = ['according to', 'research shows', 'studies indicate']
        indicators['factual_support'] = float(any(marker in text.lower() for marker in fact_markers))
        
        # Specificity
        numbers = len(re.findall(r'\d+', text))
        indicators['specificity'] = min(1.0, numbers / 10)
        
        return indicators
    
    def _analyze_domains(self, data: pd.DataFrame) -> List[DomainAnalysis]:
        """Analyze performance by domain."""
        domain_analyses = []
        
        if 'domain' not in data.columns:
            # Try to infer domains from content
            data = data.copy()
            data['domain'] = data.apply(self._infer_domain, axis=1)
        
        for domain in data['domain'].unique():
            if pd.isna(domain):
                continue
                
            domain_data = data[data['domain'] == domain]
            analysis = self._analyze_single_domain(domain, domain_data)
            domain_analyses.append(analysis)
        
        return domain_analyses
    
    def _infer_domain(self, row: pd.Series) -> str:
        """Infer domain from content."""
        text = ""
        if 'prompt' in row and pd.notna(row['prompt']):
            text += row['prompt']
        if 'response_a' in row and pd.notna(row['response_a']):
            text += " " + row['response_a']
        
        text = text.lower()
        
        # Simple domain detection
        if any(word in text for word in ['code', 'python', 'function', 'algorithm']):
            return 'programming'
        elif any(word in text for word in ['patient', 'medical', 'health', 'treatment']):
            return 'medical'
        elif any(word in text for word in ['market', 'business', 'finance', 'economy']):
            return 'business'
        elif any(word in text for word in ['history', 'historical', 'century', 'ancient']):
            return 'history'
        else:
            return 'general'
    
    def _analyze_single_domain(self, domain: str, domain_data: pd.DataFrame) -> DomainAnalysis:
        """Analyze a single domain."""
        sample_count = len(domain_data)
        
        # Calculate average quality (simplified)
        if 'chosen_index' in domain_data.columns:
            avg_quality = domain_data['chosen_index'].mean()
            quality_variance = domain_data['chosen_index'].var()
        else:
            avg_quality = 0.5
            quality_variance = 0.1
        
        # Extract common topics
        common_topics = self._extract_topics(domain_data)
        
        # Assess difficulty
        difficulty = self._assess_difficulty(domain_data)
        
        # Performance trends (simplified)
        performance_trends = {'stability': quality_variance}
        
        return DomainAnalysis(
            domain_name=domain,
            sample_count=sample_count,
            average_quality=avg_quality,
            quality_variance=quality_variance,
            common_topics=common_topics,
            difficulty_level=difficulty,
            performance_trends=performance_trends
        )
    
    def _extract_topics(self, domain_data: pd.DataFrame) -> List[str]:
        """Extract common topics from domain data."""
        # Simple keyword extraction
        all_text = ""
        for col in ['prompt', 'response_a', 'response_b']:
            if col in domain_data.columns:
                all_text += " ".join(domain_data[col].fillna("").astype(str))
        
        # Extract frequent meaningful words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        common_words = Counter(words).most_common(5)
        
        return [word for word, count in common_words if count >= 2]
    
    def _assess_difficulty(self, domain_data: pd.DataFrame) -> str:
        """Assess domain difficulty level."""
        # Simple heuristics
        if 'confidence' in domain_data.columns:
            avg_confidence = domain_data['confidence'].mean()
            if avg_confidence > 0.8:
                return 'easy'
            elif avg_confidence > 0.6:
                return 'medium'
            else:
                return 'hard'
        else:
            return 'medium'  # Default
    
    def _detect_content_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in content."""
        patterns = {}
        
        # Response length patterns
        if 'response_a' in data.columns and 'response_b' in data.columns:
            lengths_a = data['response_a'].fillna("").str.len()
            lengths_b = data['response_b'].fillna("").str.len()
            
            patterns['avg_response_length'] = {
                'response_a': lengths_a.mean(),
                'response_b': lengths_b.mean()
            }
            
            patterns['length_preference'] = 'longer' if lengths_a.mean() > lengths_b.mean() else 'shorter'
        
        # Format patterns
        if 'response_a' in data.columns:
            responses_with_lists = data['response_a'].fillna("").str.contains(r'[1-9]\.|•|\-').sum()
            patterns['structured_responses'] = responses_with_lists / len(data)
        
        return patterns
    
    def _generate_suggestions(self, quality_metrics: List[QualityMetrics], 
                            domain_analyses: List[DomainAnalysis]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if quality_metrics:
            avg_quality = np.mean([q.overall_score for q in quality_metrics])
            if avg_quality < 0.6:
                suggestions.append("📉 Low overall quality detected - focus on response improvement")
            
            avg_readability = np.mean([q.readability_score for q in quality_metrics])
            if avg_readability < 0.7:
                suggestions.append("📖 Improve response readability and structure")
        
        # Domain-specific suggestions
        low_performing_domains = [d for d in domain_analyses if d.average_quality < 0.5]
        if low_performing_domains:
            domain_names = [d.domain_name for d in low_performing_domains]
            suggestions.append(f"🎯 Focus on improving {', '.join(domain_names)} domains")
        
        if not suggestions:
            suggestions.append("✅ Content quality looks good across domains")
        
        return suggestions
    
    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Load domain-specific patterns."""
        return {
            'programming': ['code', 'function', 'algorithm', 'debug', 'syntax'],
            'medical': ['patient', 'diagnosis', 'treatment', 'symptoms', 'medicine'],
            'business': ['market', 'strategy', 'revenue', 'customer', 'profit'],
            'science': ['research', 'experiment', 'hypothesis', 'data', 'analysis']
        }
    
    def _load_quality_indicators(self) -> Dict[str, float]:
        """Load quality indicator weights."""
        return {
            'readability': 0.25,
            'coherence': 0.30,
            'completeness': 0.25,
            'accuracy': 0.20
        }


def assess_response_quality(response: str, 
                          prompt: str = "",
                          domain: str = "general") -> Dict[str, float]:
    """
    Quick quality assessment for a single response.
    
    Args:
        response: Response text to assess
        prompt: Original prompt (optional)
        domain: Content domain
        
    Returns:
        Dictionary with quality scores
    """
    analyzer = ContentAnalyzer()
    metrics = analyzer._calculate_quality_metrics(response)
    
    return {
        'overall_quality': metrics.overall_score,
        'readability': metrics.readability_score,
        'complexity': metrics.complexity_score,
        'coherence': metrics.coherence_score,
        'completeness': metrics.completeness_score
    }


def categorize_content_domains(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Automatically categorize content into domains.
    
    Args:
        data: DataFrame with content data
        
    Returns:
        Domain categorization results
    """
    analyzer = ContentAnalyzer()
    
    # Infer domains if not present
    if 'domain' not in data.columns:
        data = data.copy()
        data['domain'] = data.apply(analyzer._infer_domain, axis=1)
    
    # Analyze domain distribution
    domain_counts = data['domain'].value_counts()
    
    return {
        'domain_distribution': domain_counts.to_dict(),
        'total_domains': len(domain_counts),
        'largest_domain': domain_counts.index[0] if len(domain_counts) > 0 else None,
        'domain_balance_score': 1.0 - (domain_counts.std() / domain_counts.mean()) if len(domain_counts) > 1 else 1.0
    }