# batch_audio_analyzer.py - Interactive Batch Audio Analysis Tool
import os
import sys
import warnings

# Force Transformers to prefer local cached models
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import whisper
import numpy as np
import pandas as pd
import torch
import time
import threading
import hashlib
import json
import re
import traceback
import uuid
from datetime import datetime
from pydub import AudioSegment
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor
import pickle
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob

# Import transformers conditionally
try:
    from transformers.pipelines import pipeline as hf_pipeline
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        DistilBertTokenizerFast, 
        DistilBertForSequenceClassification,
        BertTokenizer,
        BertForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
    
    # Import PyTorch components for BiLSTM
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    PYTORCH_AVAILABLE = True
    print("Transformers library and PyTorch components loaded successfully")
    
    # GPU/CUDA Detection
    import torch
    if torch.cuda.is_available():
        DEVICE = 0  # Use first GPU
        DEVICE_NAME = f"GPU ({torch.cuda.get_device_name(0)})"
        print(f"🚀 GPU acceleration enabled: {DEVICE_NAME}")
    else:
        DEVICE = -1  # Use CPU
        DEVICE_NAME = "CPU"
        print("⚡ Using CPU processing (GPU not available)")
    
    print("Transformers library loaded successfully")
except ImportError as e:
    print(f"[WARNING] Transformers library not available: {e}")
    TRANSFORMERS_AVAILABLE = False
    PYTORCH_AVAILABLE = False
    hf_pipeline = None
    DEVICE = -1  # Default to CPU if transformers not available
    DEVICE_NAME = "CPU"

# Language detection and translation
try:
    import langdetect
    try:
        from googletrans import Translator
        GOOGLETRANS_AVAILABLE = True
    except (ImportError, AttributeError) as e:
        print(f"⚠️ Google Translate unavailable: {e}")
        GOOGLETRANS_AVAILABLE = False
        class Translator:
            def translate(self, text, dest='en'):
                return type('obj', (object,), {'text': text})
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    print("[WARNING] Language detection/translation not available.")
    MULTILINGUAL_AVAILABLE = False

# Advanced audio analysis
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    PYANNOTE_AVAILABLE = True
except ImportError:
    print("[WARNING] pyannote.audio not available. Advanced speaker diarization disabled.")
    PYANNOTE_AVAILABLE = False

# Enhanced audio processing
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    print("[WARNING] noisereduce not available. Audio noise reduction disabled.")
    NOISEREDUCE_AVAILABLE = False

warnings.filterwarnings('ignore')

# ================================
# HIPAA COMPLIANCE: PHI Scrubbing Patterns
# ================================
PHI_PATTERNS = {
    'names': re.compile(r'\b(?:my name is|i am|i\'m|call me|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE),
    'doctor_names': re.compile(r'\b(?:dr\.?|doctor|dr)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE),
    'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
    'ssn': re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
    'mrn': re.compile(r'\b(?:mrn|medical record|patient id|record number)[\s:]*([A-Z0-9]{6,12})\b', re.IGNORECASE),
    'dates': re.compile(r'\b(?:born|birth|dob|appointment|visit)[\s:]*((?:0?[1-9]|1[0-2])[\/\-\.](0?[1-9]|[12][0-9]|3[01])[\/\-\.](?:19|20)\d{2})\b', re.IGNORECASE),
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'insurance': re.compile(r'\b(?:insurance|policy|group)[\s:]*(number|#|id)[\s:]*([A-Z0-9]{6,15})\b', re.IGNORECASE),
    'addresses': re.compile(r'\b\d+\s+[A-Za-z0-9\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|court|ct|place|pl)\b', re.IGNORECASE),
    'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    'age_specific': re.compile(r'\bi am (\d{1,3}) years old\b', re.IGNORECASE)
}

# Configuration constants
HUGGINGFACE_TOKEN = "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"
TRAINING_DATA_FILE = "C:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\output_cleaned.csv"

# Download required NLTK data
import nltk
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# ================================
# OPTIMIZATION: Pre-compiled Regex Patterns
# ================================
COMPILED_PATTERNS = {
    'punctuation_numbers': re.compile(r'[^a-zA-Z\s]'),
    'whitespace': re.compile(r'\s+'),
    'negation_check': re.compile(r'\b(?:not|don\'t|no|never|none|isn\'t|wasn\'t|aren\'t|doesn\'t|didn\'t|hasn\'t|haven\'t|hadn\'t|won\'t|wouldn\'t|can\'t|couldn\'t|shouldn\'t|cannot)\b', re.IGNORECASE)
}

# ================================
# OPTIMIZATION: Multilevel Caching System
# ================================
class OptimizedCache:
    def __init__(self):
        self.max_size = 2000
        self.sentiment_cache = {}
        self.phase_cache = {}
        self.text_preprocessing_cache = {}
        self.model_prediction_cache = {}
        self.cache_lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
    def get_cache_key(self, text):
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_sentiment(self, text):
        with self.cache_lock:
            key = self.get_cache_key(text)
            result = self.sentiment_cache.get(key)
            if result is not None:
                self.hit_count += 1
            else:
                self.miss_count += 1
            return result
    
    def set_sentiment(self, text, result):
        with self.cache_lock:
            if len(self.sentiment_cache) >= self.max_size:
                # Remove oldest 20% of entries
                cleanup_count = self.max_size // 5
                to_remove = list(self.sentiment_cache.keys())[:cleanup_count]
                for k in to_remove:
                    del self.sentiment_cache[k]
            
            key = self.get_cache_key(text)
            self.sentiment_cache[key] = result
    
    def get_phase(self, text):
        with self.cache_lock:
            key = self.get_cache_key(text)
            result = self.phase_cache.get(key)
            if result is not None:
                self.hit_count += 1
            else:
                self.miss_count += 1
            return result
    
    def set_phase(self, text, result):
        with self.cache_lock:
            if len(self.phase_cache) >= self.max_size:
                cleanup_count = self.max_size // 5
                to_remove = list(self.phase_cache.keys())[:cleanup_count]
                for k in to_remove:
                    del self.phase_cache[k]
            
            key = self.get_cache_key(text)
            self.phase_cache[key] = result
    
    def get_hit_rate(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear_all(self):
        with self.cache_lock:
            self.sentiment_cache.clear()
            self.phase_cache.clear()
            self.text_preprocessing_cache.clear()
            self.model_prediction_cache.clear()
            self.hit_count = 0
            self.miss_count = 0

# Global cache instance
global_cache = OptimizedCache()

# ================================
# HIPAA COMPLIANCE: PHI Scrubbing System
# ================================
class HIPAACompliantPHIScrubber:
    def __init__(self):
        self.phi_patterns = PHI_PATTERNS
        self.scrubbed_count = 0
        self.audit_log = []
        
    def scrub_phi(self, text, preserve_clinical_context=True):
        """
        Scrub PHI from text while preserving clinical context for sentiment analysis
        """
        if not text:
            return text, 0, []
        
        original_text = text
        scrubbed_text = text
        phi_found = []
        
        # Scrub names but preserve clinical roles
        if self.phi_patterns['names'].search(scrubbed_text):
            matches = self.phi_patterns['names'].findall(scrubbed_text)
            for match in matches:
                phi_found.append(f"Patient name: {match}")
                scrubbed_text = self.phi_patterns['names'].sub(r'my name is [PATIENT_NAME]', scrubbed_text)
        
        if self.phi_patterns['doctor_names'].search(scrubbed_text):
            matches = self.phi_patterns['doctor_names'].findall(scrubbed_text)
            for match in matches:
                phi_found.append(f"Doctor name: {match}")
                scrubbed_text = self.phi_patterns['doctor_names'].sub(r'Dr. [DOCTOR_NAME]', scrubbed_text)
        
        # Scrub phone numbers
        if self.phi_patterns['phone'].search(scrubbed_text):
            phi_found.append("Phone number found")
            scrubbed_text = self.phi_patterns['phone'].sub('[PHONE_NUMBER]', scrubbed_text)
        
        # Scrub SSN
        if self.phi_patterns['ssn'].search(scrubbed_text):
            phi_found.append("SSN found")
            scrubbed_text = self.phi_patterns['ssn'].sub('[SSN]', scrubbed_text)
        
        # Scrub medical record numbers
        if self.phi_patterns['mrn'].search(scrubbed_text):
            phi_found.append("Medical record number found")
            scrubbed_text = self.phi_patterns['mrn'].sub(r'\1 [MRN]', scrubbed_text)
        
        # Scrub specific dates but preserve temporal context for sentiment analysis
        if self.phi_patterns['dates'].search(scrubbed_text):
            phi_found.append("Specific date found")
            if preserve_clinical_context:
                scrubbed_text = self.phi_patterns['dates'].sub(r'\1 [DATE]', scrubbed_text)
            else:
                scrubbed_text = self.phi_patterns['dates'].sub('[DATE]', scrubbed_text)
        
        # Scrub other PHI types
        for pattern_name in ['email', 'insurance', 'addresses', 'credit_card']:
            if self.phi_patterns[pattern_name].search(scrubbed_text):
                phi_found.append(f"{pattern_name.replace('_', ' ').title()} found")
                scrubbed_text = self.phi_patterns[pattern_name].sub(f'[{pattern_name.upper()}]', scrubbed_text)
        
        # Handle age in a privacy-preserving way
        if self.phi_patterns['age_specific'].search(scrubbed_text):
            matches = self.phi_patterns['age_specific'].findall(scrubbed_text)
            for age in matches:
                phi_found.append(f"Specific age: {age}")
                if preserve_clinical_context:
                    # Generalize age to ranges for clinical context
                    age_num = int(age)
                    if age_num < 18:
                        age_range = "minor"
                    elif age_num < 30:
                        age_range = "young adult"
                    elif age_num < 50:
                        age_range = "middle-aged adult"
                    elif age_num < 65:
                        age_range = "older adult"
                    else:
                        age_range = "senior"
                    scrubbed_text = self.phi_patterns['age_specific'].sub(f'I am a {age_range}', scrubbed_text)
                else:
                    scrubbed_text = self.phi_patterns['age_specific'].sub('I am [AGE] years old', scrubbed_text)
        
        # Update audit log
        if phi_found:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'phi_types_found': phi_found,
                'scrubbed_count': len(phi_found),
                'text_length': len(original_text)
            }
            self.audit_log.append(audit_entry)
            self.scrubbed_count += len(phi_found)
        
        return scrubbed_text, len(phi_found), phi_found
    
    def get_audit_summary(self):
        """Get HIPAA compliance audit summary"""
        return {
            'total_phi_scrubbed': self.scrubbed_count,
            'scrubbing_sessions': len(self.audit_log),
            'last_scrubbed': self.audit_log[-1]['timestamp'] if self.audit_log else None
        }

# ================================
# MULTILINGUAL SUPPORT: Language Detection and Translation
# ================================
class MultilingualAnalyzer:
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
        
    def detect_language(self, text):
        """Detect the language of input text"""
        try:
            if MULTILINGUAL_AVAILABLE:
                detected = langdetect.detect(text)
                confidence_scores = langdetect.detect_langs(text)
                confidence = next((lang.prob for lang in confidence_scores if lang.lang == detected), 0.0)
                
                return {
                    'language': detected,
                    'language_name': self.supported_languages.get(detected, 'Unknown'),
                    'confidence': confidence,
                    'supported': detected in self.supported_languages
                }
            else:
                return self._heuristic_language_detection(text)
                
        except Exception as e:
            print(f"[WARNING] Language detection failed: {e}")
            return {
                'language': 'en',
                'language_name': 'English',
                'confidence': 0.5,
                'supported': True
            }
    
    def _heuristic_language_detection(self, text):
        """Simple heuristic-based language detection as fallback"""
        text_lower = text.lower()
        
        # Common words in different languages
        language_indicators = {
            'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'está', 'dolor', 'me duele'],
            'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'pouvoir', 'douleur'],
            'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'schmerz'],
            'it': ['il', 'di', 'che', 'e', 'la', 'per', 'un', 'in', 'con', 'del', 'da', 'non', 'al', 'le', 'si', 'dei', 'come', 'io', 'questo', 'qui', 'tutto', 'ancora', 'dolore'],
            'pt': ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'dor']
        }
        
        # Count matches for each language
        scores = {}
        words = text_lower.split()
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for word in words if word in indicators)
            if words:  # Avoid division by zero
                scores[lang] = score / len(words)
            else:
                scores[lang] = 0
        
        # Default to English if no strong matches
        if not scores or max(scores.values()) < 0.1:
            detected_lang = 'en'
            confidence = 0.5
        else:
            detected_lang = max(scores, key=scores.get)
            confidence = scores[detected_lang]
        
        return {
            'language': detected_lang,
            'language_name': self.supported_languages.get(detected_lang, 'English'),
            'confidence': confidence,
            'supported': detected_lang in self.supported_languages
        }
    
    def translate_to_english(self, text, source_language):
        """Translate text to English for sentiment analysis"""
        try:
            if MULTILINGUAL_AVAILABLE:
                translator = Translator()
                result = translator.translate(text, src=source_language, dest='en')
                return {
                    'translated_text': result.text,
                    'source_language': source_language,
                    'confidence': getattr(result, 'confidence', 0.8),
                    'method': 'google_translate'
                }
            else:
                # Fallback: Use a simple mapping for common medical phrases
                return self._basic_medical_translation(text, source_language)
                
        except Exception as e:
            print(f"[WARNING] Translation failed: {e}")
            return {
                'translated_text': text,  # Return original if translation fails
                'source_language': source_language,
                'confidence': 0.0,
                'method': 'no_translation'
            }
    
    def _basic_medical_translation(self, text, source_language):
        """Basic medical phrase translation as fallback"""
        # Simple translation mappings for common medical phrases
        medical_translations = {
            'es': {
                'me duele': 'it hurts',
                'dolor': 'pain',
                'no me siento bien': 'I don\'t feel well',
                'estoy enfermo': 'I am sick',
                'tengo fiebre': 'I have fever',
                'cabeza': 'head',
                'estómago': 'stomach',
                'garganta': 'throat'
            },
            'fr': {
                'ça fait mal': 'it hurts',
                'douleur': 'pain',
                'je ne me sens pas bien': 'I don\'t feel well',
                'je suis malade': 'I am sick',
                'j\'ai de la fièvre': 'I have fever',
                'tête': 'head',
                'estomac': 'stomach',
                'gorge': 'throat'
            },
            'de': {
                'es tut weh': 'it hurts',
                'schmerz': 'pain',
                'mir geht es nicht gut': 'I don\'t feel well',
                'ich bin krank': 'I am sick',
                'ich habe fieber': 'I have fever',
                'kopf': 'head',
                'magen': 'stomach',
                'hals': 'throat'
            }
        }
        
        translated_text = text.lower()
        translation_map = medical_translations.get(source_language, {})
        
        for foreign_phrase, english_phrase in translation_map.items():
            translated_text = translated_text.replace(foreign_phrase, english_phrase)
        
        return {
            'translated_text': translated_text,
            'source_language': source_language,
            'confidence': 0.6,
            'method': 'basic_medical_mapping'
        }

# Global instances
phi_scrubber = HIPAACompliantPHIScrubber()
multilingual_analyzer = MultilingualAnalyzer()

# ================================
# BIAS DETECTION AND MITIGATION FRAMEWORK
# ================================

class DemographicBiasDetector:
    """
    Comprehensive bias detection system for gender, ethnicity, and social class
    in medical sentiment analysis
    """
    def __init__(self):
        self.bias_patterns = self._initialize_bias_patterns()
        self.bias_metrics = {
            'gender_bias_count': 0,
            'ethnicity_bias_count': 0,
            'social_class_bias_count': 0,
            'total_predictions': 0,
            'demographic_sentiment_distribution': {}
        }
        self.bias_audit_log = []
        
    def _initialize_bias_patterns(self):
        """Initialize patterns that might indicate demographic bias"""
        return {
            'gender_indicators': {
                'masculine': [
                    'he', 'his', 'him', 'man', 'male', 'gentleman', 'sir', 'father', 'dad',
                    'husband', 'boyfriend', 'son', 'brother', 'uncle', 'grandfather'
                ],
                'feminine': [
                    'she', 'her', 'hers', 'woman', 'female', 'lady', 'ma\'am', 'mother', 'mom',
                    'wife', 'girlfriend', 'daughter', 'sister', 'aunt', 'grandmother'
                ],
                'non_binary': [
                    'they', 'them', 'their', 'person', 'individual', 'patient'
                ]
            },
            'ethnicity_cultural_indicators': {
                'language_patterns': [
                    # Common patterns in different English dialects
                    'y\'all', 'ain\'t', 'finna', 'bout', 'gonna', 'wanna',  # AAVE patterns
                    'innit', 'bloke', 'bloody', 'brilliant',  # British patterns
                    'eh', 'about', 'sorry',  # Canadian patterns
                    'mate', 'reckon', 'fair dinkum'  # Australian patterns
                ],
                'cultural_medical_terms': [
                    'curandera', 'sobador', 'hierbero',  # Latino traditional healing
                    'qi', 'chi', 'acupuncture', 'herbal medicine',  # Asian traditional medicine
                    'sage', 'smudging', 'medicine wheel',  # Native American practices
                    'ayurveda', 'chakra', 'meditation'  # South Asian practices
                ]
            },
            'social_class_indicators': {
                'high_socioeconomic': [
                    'private insurance', 'concierge doctor', 'specialist', 'premium',
                    'private practice', 'executive physical', 'boutique clinic'
                ],
                'low_socioeconomic': [
                    'medicaid', 'community health', 'clinic', 'emergency room',
                    'can\'t afford', 'insurance denied', 'generic medication',
                    'free clinic', 'sliding scale'
                ],
                'education_level_high': [
                    'research shows', 'studies indicate', 'medical literature',
                    'peer-reviewed', 'clinical trial', 'evidence-based'
                ],
                'education_level_low': [
                    'heard from friend', 'saw on tv', 'read online',
                    'facebook said', 'google told me'
                ]
            }
        }
    
    def detect_demographic_indicators(self, text):
        """
        Detect potential demographic indicators in text
        """
        text_lower = text.lower()
        detected_demographics = {
            'gender': [],
            'ethnicity_culture': [],
            'social_class': [],
            'confidence_scores': {}
        }
        
        # Gender detection
        for gender_type, indicators in self.bias_patterns['gender_indicators'].items():
            matches = [indicator for indicator in indicators if indicator in text_lower]
            if matches:
                detected_demographics['gender'].append({
                    'type': gender_type,
                    'indicators': matches,
                    'confidence': len(matches) / len(indicators)
                })
        
        # Ethnicity/Cultural pattern detection
        for pattern_type, indicators in self.bias_patterns['ethnicity_cultural_indicators'].items():
            matches = [indicator for indicator in indicators if indicator in text_lower]
            if matches:
                detected_demographics['ethnicity_culture'].append({
                    'type': pattern_type,
                    'indicators': matches,
                    'confidence': len(matches) / len(indicators)
                })
        
        # Social class detection
        for class_type, indicators in self.bias_patterns['social_class_indicators'].items():
            matches = [indicator for indicator in indicators if indicator in text_lower]
            if matches:
                detected_demographics['social_class'].append({
                    'type': class_type,
                    'indicators': matches,
                    'confidence': len(matches) / len(indicators)
                })
        
        return detected_demographics
    
    def analyze_sentiment_bias(self, text, sentiment_result, demographics=None):
        """
        Analyze potential bias in sentiment prediction based on demographic indicators
        """
        if demographics is None:
            demographics = self.detect_demographic_indicators(text)
        
        bias_analysis = {
            'potential_bias_detected': False,
            'bias_types': [],
            'bias_severity': 0.0,
            'recommendations': []
        }
        
        # Check for gender bias patterns
        if demographics['gender']:
            gender_bias = self._check_gender_bias(text, sentiment_result, demographics['gender'])
            if gender_bias['bias_detected']:
                bias_analysis['potential_bias_detected'] = True
                bias_analysis['bias_types'].append('gender')
                bias_analysis['bias_severity'] += gender_bias['severity']
                bias_analysis['recommendations'].extend(gender_bias['recommendations'])
                self.bias_metrics['gender_bias_count'] += 1
        
        # Check for ethnicity/cultural bias
        if demographics['ethnicity_culture']:
            ethnicity_bias = self._check_ethnicity_bias(text, sentiment_result, demographics['ethnicity_culture'])
            if ethnicity_bias['bias_detected']:
                bias_analysis['potential_bias_detected'] = True
                bias_analysis['bias_types'].append('ethnicity_culture')
                bias_analysis['bias_severity'] += ethnicity_bias['severity']
                bias_analysis['recommendations'].extend(ethnicity_bias['recommendations'])
                self.bias_metrics['ethnicity_bias_count'] += 1
        
        # Check for social class bias
        if demographics['social_class']:
            class_bias = self._check_social_class_bias(text, sentiment_result, demographics['social_class'])
            if class_bias['bias_detected']:
                bias_analysis['potential_bias_detected'] = True
                bias_analysis['bias_types'].append('social_class')
                bias_analysis['bias_severity'] += class_bias['severity']
                bias_analysis['recommendations'].extend(class_bias['recommendations'])
                self.bias_metrics['social_class_bias_count'] += 1
        
        # Update metrics
        self.bias_metrics['total_predictions'] += 1
        
        # Log bias incident if detected
        if bias_analysis['potential_bias_detected']:
            self._log_bias_incident(text, sentiment_result, demographics, bias_analysis)
        
        return bias_analysis
    
    def _check_gender_bias(self, text, sentiment_result, gender_indicators):
        """Check for gender-based bias patterns"""
        bias_detected = False
        severity = 0.0
        recommendations = []
        
        # Example bias patterns
        text_lower = text.lower()
        
        # Check for emotional expression bias (women's pain often dismissed as emotional)
        if any(g['type'] == 'feminine' for g in gender_indicators):
            emotional_words = ['emotional', 'hysterical', 'dramatic', 'overreacting']
            if any(word in text_lower for word in emotional_words):
                if sentiment_result.get('sentiment') == 'Negative':
                    bias_detected = True
                    severity += 0.7
                    recommendations.append("Potential gender bias: Female pain may be inappropriately labeled as emotional")
        
        # Check for strength/stoicism bias (men expected to be stoic)
        if any(g['type'] == 'masculine' for g in gender_indicators):
            if 'tough it out' in text_lower or 'man up' in text_lower:
                if sentiment_result.get('sentiment') == 'Positive':
                    bias_detected = True
                    severity += 0.6
                    recommendations.append("Potential gender bias: Male pain expression may be inappropriately minimized")
        
        return {
            'bias_detected': bias_detected,
            'severity': severity,
            'recommendations': recommendations
        }
    
    def _check_ethnicity_bias(self, text, sentiment_result, ethnicity_indicators):
        """Check for ethnicity/cultural bias patterns"""
        bias_detected = False
        severity = 0.0
        recommendations = []
        
        text_lower = text.lower()
        
        # Check for language pattern bias
        for indicator in ethnicity_indicators:
            if indicator['type'] == 'language_patterns':
                # AAVE or dialect features shouldn't affect medical sentiment
                if indicator['confidence'] > 0.3:
                    # Check if sentiment is unfairly negative due to dialect
                    if sentiment_result.get('sentiment') == 'Negative' and sentiment_result.get('numerical_score', 0) < -0.5:
                        bias_detected = True
                        severity += 0.5
                        recommendations.append("Potential ethnicity bias: Dialect patterns may be affecting sentiment analysis")
        
        # Check for cultural medicine bias
        for indicator in ethnicity_indicators:
            if indicator['type'] == 'cultural_medical_terms':
                # Traditional medicine shouldn't be automatically negative
                if any(term in text_lower for term in indicator['indicators']):
                    if sentiment_result.get('sentiment') == 'Negative':
                        bias_detected = True
                        severity += 0.4
                        recommendations.append("Potential cultural bias: Traditional medicine practices may be unfairly characterized")
        
        return {
            'bias_detected': bias_detected,
            'severity': severity,
            'recommendations': recommendations
        }
    
    def _check_social_class_bias(self, text, sentiment_result, class_indicators):
        """Check for social class bias patterns"""
        bias_detected = False
        severity = 0.0
        recommendations = []
        
        # Check for healthcare access bias
        for indicator in class_indicators:
            if indicator['type'] == 'low_socioeconomic':
                # Financial constraints shouldn't make sentiment more negative
                if sentiment_result.get('sentiment') == 'Negative' and sentiment_result.get('numerical_score', 0) < -0.6:
                    bias_detected = True
                    severity += 0.6
                    recommendations.append("Potential social class bias: Financial constraints may be inappropriately affecting sentiment")
            
            elif indicator['type'] == 'high_socioeconomic':
                # Privileged healthcare access shouldn't make sentiment artificially positive
                if sentiment_result.get('sentiment') == 'Positive' and 'pain' in text.lower():
                    bias_detected = True
                    severity += 0.4
                    recommendations.append("Potential social class bias: Healthcare privilege may be masking legitimate concerns")
        
        return {
            'bias_detected': bias_detected,
            'severity': severity,
            'recommendations': recommendations
        }
    
    def _log_bias_incident(self, text, sentiment_result, demographics, bias_analysis):
        """Log bias incident for audit purposes"""
        incident = {
            'timestamp': datetime.now().isoformat(),
            'text_sample': text[:100] + "..." if len(text) > 100 else text,
            'sentiment_result': sentiment_result,
            'demographics_detected': demographics,
            'bias_analysis': bias_analysis,
            'bias_severity': bias_analysis['bias_severity']
        }
        self.bias_audit_log.append(incident)
        
        # Keep only last 1000 incidents
        if len(self.bias_audit_log) > 1000:
            self.bias_audit_log = self.bias_audit_log[-1000:]
    
    def get_bias_metrics(self):
        """Get comprehensive bias metrics"""
        total_predictions = max(1, self.bias_metrics['total_predictions'])
        
        return {
            'bias_detection_rate': {
                'gender_bias_rate': self.bias_metrics['gender_bias_count'] / total_predictions,
                'ethnicity_bias_rate': self.bias_metrics['ethnicity_bias_count'] / total_predictions,
                'social_class_bias_rate': self.bias_metrics['social_class_bias_count'] / total_predictions,
                'overall_bias_rate': (
                    self.bias_metrics['gender_bias_count'] + 
                    self.bias_metrics['ethnicity_bias_count'] + 
                    self.bias_metrics['social_class_bias_count']
                ) / total_predictions
            },
            'total_predictions': total_predictions,
            'recent_incidents': len([
                incident for incident in self.bias_audit_log 
                if (datetime.now() - datetime.fromisoformat(incident['timestamp'])).days < 7
            ])
        }
    
    def generate_bias_report(self):
        """Generate comprehensive bias analysis report"""
        metrics = self.get_bias_metrics()
        
        return {
            'summary': {
                'overall_bias_rate': metrics['bias_detection_rate']['overall_bias_rate'],
                'most_common_bias_type': max(
                    [
                        ('gender', metrics['bias_detection_rate']['gender_bias_rate']),
                        ('ethnicity', metrics['bias_detection_rate']['ethnicity_bias_rate']),
                        ('social_class', metrics['bias_detection_rate']['social_class_bias_rate'])
                    ],
                    key=lambda x: x[1]
                )[0],
                'total_incidents': len(self.bias_audit_log)
            },
            'detailed_metrics': metrics,
            'recent_high_severity_incidents': [
                incident for incident in self.bias_audit_log[-50:]
                if incident['bias_severity'] > 0.7
            ],
            'recommendations': self._generate_mitigation_recommendations(metrics)
        }
    
    def _generate_mitigation_recommendations(self, metrics):
        """Generate recommendations for bias mitigation"""
        recommendations = []
        
        if metrics['bias_detection_rate']['gender_bias_rate'] > 0.05:
            recommendations.append({
                'type': 'gender_bias_mitigation',
                'priority': 'high',
                'action': 'Implement gender-neutral preprocessing and bias-aware training data augmentation'
            })
        
        if metrics['bias_detection_rate']['ethnicity_bias_rate'] > 0.03:
            recommendations.append({
                'type': 'ethnicity_bias_mitigation',
                'priority': 'high',
                'action': 'Add cultural sensitivity training data and dialect normalization'
            })
        
        if metrics['bias_detection_rate']['social_class_bias_rate'] > 0.04:
            recommendations.append({
                'type': 'social_class_bias_mitigation',
                'priority': 'medium',
                'action': 'Implement socioeconomic-aware context understanding and bias correction'
            })
        
        return recommendations

class BiasMitigationProcessor:
    """
    Preprocessing and post-processing to reduce demographic bias
    """
    def __init__(self):
        self.bias_detector = DemographicBiasDetector()
        self.normalization_patterns = self._initialize_normalization_patterns()
        
    def _initialize_normalization_patterns(self):
        """Initialize patterns for demographic normalization"""
        return {
            'gender_neutralization': {
                # Replace gendered terms with neutral equivalents
                r'\bhe is\b': 'the patient is',
                r'\bshe is\b': 'the patient is',
                r'\bhis\b': 'their',
                r'\bher\b': 'their',
                r'\bhim\b': 'them',
                r'\bman\b': 'person',
                r'\bwoman\b': 'person',
                r'\bmale patient\b': 'patient',
                r'\bfemale patient\b': 'patient'
            },
            'dialect_standardization': {
                # Standardize dialect variations (carefully to preserve meaning)
                r'\bain\'t\b': 'is not',
                r'\bgonna\b': 'going to',
                r'\bwanna\b': 'want to',
                r'\bfinna\b': 'about to',
                r'\by\'all\b': 'you all'
            },
            'socioeconomic_neutralization': {
                # Neutralize socioeconomic indicators where appropriate
                r'\bprivate insurance\b': 'insurance',
                r'\bmedicaid\b': 'insurance',
                r'\bfree clinic\b': 'clinic',
                r'\bprivate practice\b': 'practice',
                r'\bcommunity health center\b': 'health center'
            }
        }
    
    def preprocess_for_bias_reduction(self, text, apply_gender_neutralization=True, 
                                    apply_dialect_standardization=True, 
                                    apply_socioeconomic_neutralization=False):
        """
        Apply bias-reducing preprocessing to text
        """
        processed_text = text
        applied_transformations = []
        
        if apply_gender_neutralization:
            for pattern, replacement in self.normalization_patterns['gender_neutralization'].items():
                if re.search(pattern, processed_text, re.IGNORECASE):
                    processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
                    applied_transformations.append(f"Gender neutralization: {pattern} -> {replacement}")
        
        if apply_dialect_standardization:
            for pattern, replacement in self.normalization_patterns['dialect_standardization'].items():
                if re.search(pattern, processed_text, re.IGNORECASE):
                    processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
                    applied_transformations.append(f"Dialect standardization: {pattern} -> {replacement}")
        
        if apply_socioeconomic_neutralization:
            for pattern, replacement in self.normalization_patterns['socioeconomic_neutralization'].items():
                if re.search(pattern, processed_text, re.IGNORECASE):
                    processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
                    applied_transformations.append(f"Socioeconomic neutralization: {pattern} -> {replacement}")
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'transformations_applied': applied_transformations,
            'bias_reduction_level': len(applied_transformations)
        }
    
    def postprocess_sentiment_with_bias_correction(self, text, sentiment_result):
        """
        Apply bias correction to sentiment results
        """
        # Detect demographic indicators
        demographics = self.bias_detector.detect_demographic_indicators(text)
        
        # Analyze potential bias
        bias_analysis = self.bias_detector.analyze_sentiment_bias(text, sentiment_result, demographics)
        
        corrected_result = sentiment_result.copy()
        
        # Apply bias corrections based on detected patterns
        if bias_analysis['potential_bias_detected']:
            correction_applied = False
            
            # Gender bias correction
            if 'gender' in bias_analysis['bias_types']:
                if bias_analysis['bias_severity'] > 0.6:
                    # Moderate the sentiment score to reduce bias
                    if 'numerical_score' in corrected_result:
                        original_score = corrected_result['numerical_score']
                        corrected_result['numerical_score'] = original_score * 0.8  # Reduce intensity
                        corrected_result['bias_correction'] = 'gender_bias_moderation'
                        correction_applied = True
            
            # Ethnicity bias correction
            if 'ethnicity_culture' in bias_analysis['bias_types']:
                if bias_analysis['bias_severity'] > 0.5:
                    # Shift slightly toward neutral
                    if 'numerical_score' in corrected_result:
                        original_score = corrected_result['numerical_score']
                        corrected_result['numerical_score'] = original_score * 0.7
                        corrected_result['bias_correction'] = 'ethnicity_bias_moderation'
                        correction_applied = True
            
            # Social class bias correction
            if 'social_class' in bias_analysis['bias_types']:
                if bias_analysis['bias_severity'] > 0.6:
                    # Adjust sentiment to account for socioeconomic context
                    if 'numerical_score' in corrected_result:
                        original_score = corrected_result['numerical_score']
                        corrected_result['numerical_score'] = original_score * 0.75
                        corrected_result['bias_correction'] = 'social_class_bias_moderation'
                        correction_applied = True
            
            if correction_applied:
                # Recalculate sentiment label based on corrected score
                score = corrected_result.get('numerical_score', 0)
                if score < -0.1:
                    corrected_result['sentiment'] = 'Negative'
                elif score > 0.1:
                    corrected_result['sentiment'] = 'Positive'
                else:
                    corrected_result['sentiment'] = 'Neutral'
        
        # Add bias analysis to result
        corrected_result['bias_analysis'] = bias_analysis
        corrected_result['demographics_detected'] = demographics
        
        return corrected_result

# Global bias detection and mitigation instances
bias_detector = DemographicBiasDetector()
bias_mitigator = BiasMitigationProcessor()

# ================================
# OPTIMIZATION: Performance Decorators
# ================================
def cache_result(cache_type='sentiment'):
    def decorator(func):
        @wraps(func)
        def wrapper(self, text, *args, **kwargs):
            if cache_type == 'sentiment':
                cached = global_cache.get_sentiment(text)
                if cached is not None:
                    return cached
            elif cache_type == 'phase':
                cached = global_cache.get_phase(text)
                if cached is not None:
                    return cached
                    
            result = func(self, text, *args, **kwargs)
            
            if cache_type == 'sentiment':
                global_cache.set_sentiment(text, result)
            elif cache_type == 'phase':
                global_cache.set_phase(text, result)
                
            return result
        return wrapper
    return decorator

def batch_process(batch_size=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, texts, *args, **kwargs):
            if isinstance(texts, str):
                return func(self, texts, *args, **kwargs)
            
            effective_batch_size = batch_size or 16
            results = []
            
            for i in range(0, len(texts), effective_batch_size):
                batch = texts[i:i + effective_batch_size]
                batch_results = [func(self, text, *args, **kwargs) for text in batch]
                results.extend(batch_results)
                    
            return results
        return wrapper
    return decorator

# ================================
# OPTIMIZATION: Model Loading with Lazy Initialization
# ================================
class ModelManager:
    def __init__(self):
        self._models = {}
        self._loading_lock = threading.RLock()
        self._diarization_pipeline = None
        self._whisper_model = None
        self._speaker_classifier = None
        
    def get_diarization_pipeline(self):
        if self._diarization_pipeline is None:
            with self._loading_lock:
                if self._diarization_pipeline is None:
                    print(f"[{DEVICE_NAME}] Loading diarization pipeline...")
                    try:
                        if PYANNOTE_AVAILABLE:
                            self._diarization_pipeline = Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.1",
                                use_auth_token=HUGGINGFACE_TOKEN
                            )
                            # Move to GPU if available
                            if DEVICE != -1:
                                self._diarization_pipeline = self._diarization_pipeline.to(torch.device(f"cuda:{DEVICE}"))
                            print(f"[{DEVICE_NAME}] Diarization pipeline loaded on {DEVICE_NAME}")
                        else:
                            print("[WARNING] pyannote.audio not available")
                            self._diarization_pipeline = None
                    except Exception as e:
                        print(f"[WARNING] Could not load diarization pipeline: {e}")
                        self._diarization_pipeline = None
        return self._diarization_pipeline
    
    def get_whisper_model(self):
        if self._whisper_model is None:
            with self._loading_lock:
                if self._whisper_model is None:
                    print(f"[{DEVICE_NAME}] Loading Whisper model from cache...")
                    # Use base model for faster loading and lower memory usage
                    model_size = 'base'
                    
                    try:
                        # Check if model exists in cache
                        cache_dir = os.path.expanduser("~/.cache/whisper")
                        model_path = os.path.join(cache_dir, f"{model_size}.pt")
                        
                        if os.path.exists(model_path):
                            print(f"[{DEVICE_NAME}] Using cached Whisper model: {model_path}")
                            # Set download root to cache directory to prevent downloading
                            os.environ['WHISPER_CACHE_DIR'] = cache_dir
                            self._whisper_model = whisper.load_model(model_size, download_root=cache_dir)
                            print(f"[{DEVICE_NAME}] Successfully loaded cached Whisper model ({model_size}) on {DEVICE_NAME}")
                        else:
                            print(f"[{DEVICE_NAME}] Cache miss for {model_size}, checking for alternatives...")
                            # Try other available cached models
                            available_models = []
                            for check_model in ['base', 'tiny', 'small']:
                                check_path = os.path.join(cache_dir, f"{check_model}.pt")
                                if os.path.exists(check_path):
                                    available_models.append(check_model)
                            
                            if available_models:
                                fallback_model = available_models[0]  # Use first available
                                print(f"[{DEVICE_NAME}] Using available cached model: {fallback_model}")
                                os.environ['WHISPER_CACHE_DIR'] = cache_dir
                                self._whisper_model = whisper.load_model(fallback_model, download_root=cache_dir)
                                print(f"[{DEVICE_NAME}] Successfully loaded cached Whisper model ({fallback_model}) on {DEVICE_NAME}")
                            else:
                                print(f"[{DEVICE_NAME}] No cached models found, loading {model_size}...")
                                self._whisper_model = whisper.load_model(model_size)
                                print(f"[{DEVICE_NAME}] Whisper model ({model_size}) loaded on {DEVICE_NAME}")
                            
                    except Exception as e:
                        print(f"[WARNING] Error loading cached Whisper model: {e}")
                        print(f"[{DEVICE_NAME}] Falling back to basic Whisper loading...")
                        try:
                            self._whisper_model = whisper.load_model('tiny')  # Smallest model as final fallback
                            print(f"[{DEVICE_NAME}] Fallback: Whisper tiny model loaded on {DEVICE_NAME}")
                        except Exception as e2:
                            print(f"[ERROR] Failed to load any Whisper model: {e2}")
                            self._whisper_model = None
                        
        return self._whisper_model
    
    def get_speaker_classifier(self):
        """Load the fine-tuned DistilBERT speaker classifier"""
        if self._speaker_classifier is None:
            with self._loading_lock:
                if self._speaker_classifier is None:
                    print("[SPEAKER CLASSIFIER] Loading fine-tuned DistilBERT speaker classifier...")
                    try:
                        if TRANSFORMERS_AVAILABLE:
                            model_path = "./distilbert-finetuned-unlabeled"
                            
                            # Load the fine-tuned model and tokenizer
                            tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
                            model = DistilBertForSequenceClassification.from_pretrained(model_path)
                            
                            # Create pipeline for classification
                            self._speaker_classifier = hf_pipeline(
                                "text-classification",
                                model=model,
                                tokenizer=tokenizer,
                                device=DEVICE,  # Use GPU if available
                                return_all_scores=True
                            )
                            print("[SPEAKER CLASSIFIER] Fine-tuned DistilBERT speaker classifier loaded successfully")
                        else:
                            print("[WARNING] Transformers not available for speaker classification")
                            self._speaker_classifier = None
                    except Exception as e:
                        print(f"[WARNING] Could not load speaker classifier: {e}")
                        self._speaker_classifier = None
        return self._speaker_classifier

# Global model manager
model_manager = ModelManager()

# ================================
# BILSTM CONTEXTUAL SENTIMENT ANALYZER
# ================================

class ConversationContextEncoder(nn.Module):
    """
    BiLSTM-based contextual encoder that maintains conversation history
    for improved sentiment analysis in medical conversations
    """
    def __init__(self, embedding_dim=512, hidden_dim=256, num_layers=2, dropout=0.3):
        super(ConversationContextEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM for conversation context modeling
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism for context weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Context fusion layers
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # Negative, Neutral, Positive
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, embeddings, conversation_history=None):
        """
        Enhanced forward pass through the contextual encoder with attention to question-answer patterns
        
        Args:
            embeddings: Current sentence embeddings [batch_size, embedding_dim]
            conversation_history: Previous conversation embeddings [batch_size, seq_len, embedding_dim]
        
        Returns:
            contextual_sentiment: Sentiment logits considering conversation context with Q&A understanding
        """
        batch_size = embeddings.size(0)
        
        if conversation_history is not None and conversation_history.size(1) > 0:
            # Combine current embeddings with conversation history
            # conversation_history: [batch_size, seq_len, embedding_dim]
            # embeddings: [batch_size, embedding_dim] -> [batch_size, 1, embedding_dim]
            current_embedding = embeddings.unsqueeze(1)
            
            # Concatenate with conversation history
            full_sequence = torch.cat([conversation_history, current_embedding], dim=1)
            
            # Pass through BiLSTM to capture sequential dependencies
            lstm_output, (hidden, cell) = self.bilstm(full_sequence)
            
            # Apply layer normalization
            lstm_output = self.layer_norm(lstm_output)
            
            # Enhanced attention mechanism for question-answer understanding
            # Give higher attention to the most recent exchange (likely question-answer pair)
            seq_len = lstm_output.size(1)
            
            # Create position weights that emphasize recent context
            position_weights = torch.linspace(0.1, 1.0, seq_len).to(lstm_output.device)
            position_weights = position_weights.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1]
            
            # Weight the LSTM outputs by position (more recent = higher weight)
            weighted_lstm_output = lstm_output * position_weights
            
            # Apply attention to focus on relevant context
            attended_output, attention_weights = self.attention(
                weighted_lstm_output, weighted_lstm_output, weighted_lstm_output
            )
            
            # Enhanced context fusion: Consider both immediate previous turn and overall context
            # Get the current sentence representation (last in sequence)
            current_context = attended_output[:, -1, :]  # [batch_size, hidden_dim * 2]
            
            # Also get the immediate previous context (likely the question if this is an answer)
            if seq_len > 1:
                previous_context = attended_output[:, -2, :]  # [batch_size, hidden_dim * 2]
                
                # Create a weighted combination emphasizing question-answer relationship
                immediate_context = 0.7 * previous_context + 0.3 * current_context
            else:
                immediate_context = current_context
            
            # Fuse with original embeddings for comprehensive representation
            fused_features = torch.cat([immediate_context, embeddings], dim=1)
            
        else:
            # No conversation history, use only current embeddings
            # Create a simple context representation for standalone analysis
            current_embedding = embeddings.unsqueeze(1)
            lstm_output, _ = self.bilstm(current_embedding)
            current_context = lstm_output[:, -1, :]
            
            # Fuse with original embeddings
            fused_features = torch.cat([current_context, embeddings], dim=1)
        
        # Generate contextual sentiment prediction with enhanced understanding
        contextual_sentiment = self.context_fusion(fused_features)
        
        return contextual_sentiment

class BiLSTMContextualSentimentAnalyzer:
    """
    Enhanced sentiment analyzer that combines XLM-RoBERTa with BiLSTM
    for conversation context understanding
    """
    def __init__(self, xlm_roberta_pipeline, device='cpu'):
        self.xlm_roberta = xlm_roberta_pipeline
        self.device = device
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 conversation turns
        
        # Initialize BiLSTM contextual encoder
        if PYTORCH_AVAILABLE:
            self.context_encoder = ConversationContextEncoder()
            if device != 'cpu' and torch.cuda.is_available():
                self.context_encoder = self.context_encoder.to(device)
            self.context_encoder.eval()  # Set to evaluation mode
        else:
            self.context_encoder = None
            print("[WARNING] PyTorch not available. Using basic XLM-RoBERTa without contextual enhancement.")
    
    def get_xlm_roberta_embeddings(self, text):
        """
        Extract embeddings from XLM-RoBERTa model
        """
        try:
            if hasattr(self.xlm_roberta, 'model') and hasattr(self.xlm_roberta, 'tokenizer'):
                # Tokenize and get model outputs
                inputs = self.xlm_roberta.tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                if self.device != 'cpu':
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.xlm_roberta.model(**inputs)
                    # Check if we have hidden states or different output structure
                    if hasattr(outputs, 'last_hidden_state'):
                        # Use [CLS] token embeddings as sentence representation
                        embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
                    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        # Use last layer hidden states
                        embeddings = outputs.hidden_states[-1][:, 0, :]
                    elif hasattr(outputs, 'logits'):
                        # Fallback: use logits as crude embeddings
                        embeddings = outputs.logits
                    else:
                        # Fallback: create dummy embeddings
                        embeddings = torch.randn(1, 512)
                
                # Ensure embeddings are in float32 format for BiLSTM compatibility
                embeddings = embeddings.to(torch.float32)
                return embeddings
            else:
                # Fallback: create dummy embeddings in float32
                return torch.randn(1, 512, dtype=torch.float32)
                
        except Exception as e:
            print(f"[WARNING] Error extracting XLM-RoBERTa embeddings: {e}")
            # Return dummy embeddings as fallback in float32
            return torch.randn(1, 512, dtype=torch.float32)
    
    def add_to_conversation_history(self, text, speaker='PATIENT'):
        """
        Add a conversation turn to the history with context management
        """
        # Get embeddings for the text
        embeddings = self.get_xlm_roberta_embeddings(text)
        
        # Add to conversation history
        conversation_turn = {
            'text': text,
            'speaker': speaker,
            'embeddings': embeddings,
            'timestamp': datetime.now()
        }
        
        self.conversation_history.append(conversation_turn)
        
        # Maintain maximum history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_conversation_context_embeddings(self):
        """
        Get embeddings for conversation history
        """
        if not self.conversation_history:
            return None
        
        # Stack embeddings from conversation history
        history_embeddings = torch.cat([
            turn['embeddings'] for turn in self.conversation_history[:-1]  # Exclude current turn
        ], dim=0).unsqueeze(0)  # Add batch dimension
        
        return history_embeddings
    
    def analyze_contextual_sentiment(self, text, speaker='PATIENT'):
        """
        Enhanced contextual sentiment analysis with medical question-answer understanding
        """
        # Step 1: Apply bias-reducing preprocessing
        preprocessed = bias_mitigator.preprocess_for_bias_reduction(
            text, 
            apply_gender_neutralization=False,  # Keep gender info for medical context
            apply_dialect_standardization=True,
            apply_socioeconomic_neutralization=False  # Keep socioeconomic context for healthcare access understanding
        )
        
        # Use original text for analysis but note preprocessing was applied
        analysis_text = text  # We'll use bias correction in post-processing instead
        
        # Add current text to conversation history
        self.add_to_conversation_history(analysis_text, speaker)
        
        # Check for medical question-answer patterns before sentiment analysis
        medical_context = self.analyze_medical_question_answer_context(text, speaker)
        
        # Get basic XLM-RoBERTa sentiment (fallback)
        try:
            basic_sentiment = self.xlm_roberta(analysis_text)[0]
            basic_score = basic_sentiment['score']
            basic_label = basic_sentiment['label']
            
            # Map to standardized format
            if basic_label.upper() in ['NEGATIVE', 'NEG']:
                sentiment_label = 'Negative'
                numerical_score = -basic_score
            elif basic_label.upper() in ['POSITIVE', 'POS']:
                sentiment_label = 'Positive'
                numerical_score = basic_score
            else:
                sentiment_label = 'Neutral'
                numerical_score = 0.0
                
        except Exception as e:
            print(f"[WARNING] XLM-RoBERTa sentiment analysis failed: {e}")
            sentiment_label = 'Neutral'
            numerical_score = 0.0
            basic_score = 0.5
        
        # Apply medical context override if detected
        if medical_context['override_needed']:
            print(f"[MEDICAL CONTEXT] Applying override: {medical_context['reason']}")
            result = {
                'sentiment': medical_context['corrected_sentiment'],
                'numerical_score': medical_context['corrected_score'],
                'confidence': medical_context['confidence'],
                'method': 'medical_context_override',
                'original_sentiment': sentiment_label,
                'override_reason': medical_context['reason'],
                'text': text,
                'preprocessing_applied': preprocessed['transformations_applied']
            }
            
            # Apply bias detection and correction
            bias_corrected_result = bias_mitigator.postprocess_sentiment_with_bias_correction(
                text, result
            )
            
            return bias_corrected_result
        
        # Enhanced contextual analysis if BiLSTM is available
        if self.context_encoder is not None and len(self.conversation_history) > 1:
            try:
                # Get current sentence embeddings
                current_embeddings = self.conversation_history[-1]['embeddings']
                
                # Get conversation context
                context_embeddings = self.get_conversation_context_embeddings()
                
                # Apply contextual analysis
                with torch.no_grad():
                    contextual_logits = self.context_encoder(current_embeddings, context_embeddings)
                    contextual_probs = F.softmax(contextual_logits, dim=1)
                    
                    # Get predictions
                    predicted_class = torch.argmax(contextual_probs, dim=1).item()
                    confidence = torch.max(contextual_probs, dim=1)[0].item()
                    
                    # Map predictions to sentiment labels
                    if predicted_class == 0:  # Negative
                        contextual_label = 'Negative'
                        contextual_score = -confidence
                    elif predicted_class == 2:  # Positive
                        contextual_label = 'Positive'
                        contextual_score = confidence
                    else:  # Neutral
                        contextual_label = 'Neutral'
                        contextual_score = 0.0
                
                # Apply medical context corrections
                corrected_sentiment = self.apply_medical_context_corrections(
                    analysis_text, contextual_label, contextual_score, speaker
                )
                
                # Create contextual result
                result = {
                    'sentiment': corrected_sentiment['sentiment'],
                    'numerical_score': corrected_sentiment['score'],
                    'confidence': confidence,
                    'method': 'bilstm_contextual',
                    'basic_sentiment': sentiment_label,
                    'basic_score': basic_score,
                    'contextual_correction': corrected_sentiment['correction'],
                    'correction_context': corrected_sentiment.get('context', 'N/A'),
                    'text': text,
                    'preprocessing_applied': preprocessed['transformations_applied']
                }
                
                # Apply bias detection and correction
                bias_corrected_result = bias_mitigator.postprocess_sentiment_with_bias_correction(
                    text, result
                )
                
                return bias_corrected_result
                
            except Exception as e:
                print(f"[WARNING] Contextual analysis failed: {e}")
                # Fall back to basic sentiment
                pass
        
        # Apply basic medical context corrections
        corrected_sentiment = self.apply_medical_context_corrections(
            analysis_text, sentiment_label, numerical_score, speaker
        )
        
        # Create basic result
        result = {
            'sentiment': corrected_sentiment['sentiment'],
            'numerical_score': corrected_sentiment['score'],
            'confidence': basic_score,
            'method': 'xlm_roberta_basic',
            'contextual_correction': corrected_sentiment['correction'],
            'correction_context': corrected_sentiment.get('context', 'N/A'),
            'text': text,
            'preprocessing_applied': preprocessed['transformations_applied']
        }
        
        # Apply bias detection and correction even for basic analysis
        bias_corrected_result = bias_mitigator.postprocess_sentiment_with_bias_correction(
            text, result
        )
        
        return bias_corrected_result
    
    def analyze_medical_question_answer_context(self, text, speaker):
        """
        Analyze if current text is part of a medical question-answer pattern
        Returns context information that may override standard sentiment analysis
        """
        text_lower = text.lower().strip()
        
        # Initialize result
        context_result = {
            'override_needed': False,
            'corrected_sentiment': None,
            'corrected_score': 0.0,
            'confidence': 0.5,
            'reason': 'no_override',
            'pattern': None
        }
        
        # Only analyze patient responses for Q&A context
        if speaker != 'PATIENT' or len(self.conversation_history) < 2:
            return context_result
        
        # Get the previous turn (likely doctor's question)
        previous_turn = self.conversation_history[-2]['text'].lower()
        
        # Check for uncertainty indicators first (higher priority than simple negation)
        uncertainty_patterns = [
            r'not sure', r'maybe', r'perhaps', r'possibly', r'might', r'could be',
            r'i think', r'i guess', r'probably', r'sometimes', r'occasionally',
            r'a little', r'little bit', r'sort of', r'kind of'
        ]
        
        has_uncertainty = any(re.search(pattern, text_lower) for pattern in uncertainty_patterns)
        
        # Define medical question patterns and their expected positive/negative answers
        medical_qa_patterns = {
            'symptom_inquiry': {
                'question_keywords': ['fever', 'pain', 'hurt', 'ache', 'nausea', 'sick', 'tired', 
                                    'cough', 'breathing', 'dizzy', 'headache', 'stomach', 'problem',
                                    'trouble', 'difficulty', 'experiencing'],
                'positive_answers': {  # Answers that indicate NO symptoms (good health outcome)
                    'negation': [r'\bno\b', r'\bnot\b', r"don't\b", r"doesn't\b", r'\bnever\b'],
                    'denial': ['fine', 'good', 'okay', 'ok', 'normal', 'well'],
                    'improvement': ['better', 'improved', 'recovering']
                },
                'negative_answers': {  # Answers that confirm symptoms (health concerns)
                    'confirmation': [r'\byes\b', r'\byeah\b', r'\bsure\b', r'\bdefinitely\b'],
                    'severity': ['bad', 'worse', 'terrible', 'awful', 'severe']
                }
            },
            'wellbeing_inquiry': {
                'question_keywords': ['feeling', 'how are you', 'doing', 'today', 'condition'],
                'positive_answers': {
                    'good_state': ['good', 'fine', 'well', 'better', 'okay', 'ok', 'great'],
                    'improvement': ['improving', 'recovering', 'healing']
                },
                'negative_answers': {
                    'poor_state': ['bad', 'terrible', 'awful', 'worse', 'sick', 'tired']
                }
            }
        }
        
        # Check each pattern
        for pattern_name, pattern_data in medical_qa_patterns.items():
            # Check if previous turn contains question keywords
            question_detected = any(
                keyword in previous_turn for keyword in pattern_data['question_keywords']
            )
            
            if question_detected:
                # Special handling for uncertainty (should be neutral)
                if has_uncertainty:
                    context_result.update({
                        'override_needed': True,
                        'corrected_sentiment': 'Neutral',
                        'corrected_score': 0.1,  # Slight positive bias
                        'confidence': 0.7,
                        'reason': 'medical_qa_uncertainty',
                        'pattern': pattern_name,
                        'detected_pattern': 'uncertainty_expression',
                        'question_context': previous_turn[:100]
                    })
                    return context_result
                
                # Check for positive answer patterns (should result in positive sentiment)
                if 'positive_answers' in pattern_data:
                    for answer_type, patterns in pattern_data['positive_answers'].items():
                        for pattern in patterns:
                            if re.search(pattern, text_lower):
                                context_result.update({
                                    'override_needed': True,
                                    'corrected_sentiment': 'Positive',
                                    'corrected_score': 0.7,
                                    'confidence': 0.8,
                                    'reason': f'medical_qa_positive_{answer_type}',
                                    'pattern': pattern_name,
                                    'detected_pattern': pattern,
                                    'question_context': previous_turn[:100]
                                })
                                return context_result
                
                # Check for negative answer patterns (should result in negative sentiment)
                if 'negative_answers' in pattern_data:
                    for answer_type, patterns in pattern_data['negative_answers'].items():
                        for pattern in patterns:
                            if re.search(pattern, text_lower):
                                context_result.update({
                                    'override_needed': True,
                                    'corrected_sentiment': 'Negative',
                                    'corrected_score': -0.6,
                                    'confidence': 0.8,
                                    'reason': f'medical_qa_negative_{answer_type}',
                                    'pattern': pattern_name,
                                    'detected_pattern': pattern,
                                    'question_context': previous_turn[:100]
                                })
                                return context_result
        
        return context_result
    
    def apply_medical_context_corrections(self, text, sentiment_label, numerical_score, speaker):
        """
        Apply advanced medical context-specific corrections to sentiment analysis with BiLSTM contextual understanding
        """
        text_lower = text.lower().strip()
        
        # Enhanced medical symptom indicators  
        symptom_indicators = {
            'pain': ['pain', 'hurt', 'ache', 'aching', 'sore', 'tender'],
            'severity': ['sharp', 'burning', 'stabbing', 'throbbing', 'constant', 'severe', 'chronic'],
            'discomfort': ['uncomfortable', 'discomfort', 'bothering', 'irritating'],
            'respiratory': ['fever', 'cough', 'breathing', 'shortness', 'wheezing'],
            'gastrointestinal': ['nausea', 'vomiting', 'stomach', 'digestion', 'appetite'],
            'general': ['tired', 'fatigue', 'weakness', 'dizzy', 'headache']
        }
        
        # Negation patterns - more comprehensive
        negation_patterns = [
            r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bnothing\b', r'\bnobody\b',
            r'\bnone\b', r'\bneither\b', r'\bnor\b', r'\bdont\b', r"don't\b",
            r'\bdidnt\b', r"didn't\b", r'\bwont\b', r"won't\b", r'\bcant\b',
            r"can't\b", r'\bwouldnt\b', r"wouldn't\b", r'\bshouldnt\b', r"shouldn't\b"
        ]
        
        # Question keywords that typically indicate doctor queries about symptoms
        medical_question_keywords = [
            'fever', 'pain', 'hurt', 'ache', 'feeling', 'symptoms', 'problems',
            'issues', 'concerns', 'bothering', 'trouble', 'difficulty', 'experience',
            'notice', 'changes', 'different', 'worse', 'better'
        ]
        
        # Get recent conversation context (last 3 turns for better context)
        recent_context = []
        if len(self.conversation_history) >= 2:
            # Look at previous turn (likely doctor's question)
            previous_turn = self.conversation_history[-2]['text'].lower()
            recent_context.append(previous_turn)
            
            # Also look at 2 turns back for additional context
            if len(self.conversation_history) >= 3:
                recent_context.append(self.conversation_history[-3]['text'].lower())
        
        combined_context = ' '.join(recent_context)
        
        # Check if this is a patient response
        if speaker == 'PATIENT':
            
            # Pattern 1: NEGATION OF SYMPTOMS (should be POSITIVE for health outcomes)
            # E.g., "No, I don't think so" after "Do you have a fever?"
            has_negation = any(re.search(pattern, text_lower) for pattern in negation_patterns)
            
            if has_negation:
                # Check if recent context contains medical questions about symptoms
                context_has_symptom_question = any(
                    keyword in combined_context for keyword in medical_question_keywords
                )
                
                # Check if the negation is about symptoms/problems
                symptom_context = any(
                    any(symptom in combined_context for symptom in category_symptoms)
                    for category_symptoms in symptom_indicators.values()
                )
                
                # CONTEXTUAL CORRECTION: Negating symptoms should be positive
                if context_has_symptom_question or symptom_context:
                    return {
                        'sentiment': 'Positive',
                        'score': abs(numerical_score) if numerical_score != 0 else 0.6,
                        'correction': 'contextual_symptom_negation',
                        'context': 'Patient denying symptoms/problems - positive health outcome'
                    }
            
            # Pattern 2: CONFIRMATION OF SYMPTOMS (should be NEGATIVE)
            # E.g., "Yes, it hurts" or "Yeah, definitely"
            confirmation_words = [
                r'\byes\b', r'\byeah\b', r'\byep\b', r'\buh-huh\b',
                r'\bdefinitely\b', r'\babsolutely\b', r'\bcorrect\b', r'\bright\b',
                r'\bexactly\b', r'\bsure\b', r'\bcertainly\b', r'\bindeed\b'
            ]
            
            has_confirmation = any(re.search(pattern, text_lower) for pattern in confirmation_words)
            
            if has_confirmation:
                # Check if confirming symptoms/pain
                symptom_context = any(
                    any(symptom in combined_context for symptom in category_symptoms)
                    for category_symptoms in symptom_indicators.values()
                )
                
                if symptom_context:
                    return {
                        'sentiment': 'Negative',
                        'score': -abs(numerical_score) if numerical_score != 0 else -0.6,
                        'correction': 'contextual_symptom_confirmation',
                        'context': 'Patient confirming symptoms/problems'
                    }
            
            # Pattern 3: DIRECT SYMPTOM DESCRIPTION
            # Check for direct mentions of symptoms in patient response
            patient_describes_symptoms = any(
                any(symptom in text_lower for symptom in category_symptoms)
                for category_symptoms in symptom_indicators.values()
            )
            
            if patient_describes_symptoms and sentiment_label != 'Negative':
                return {
                    'sentiment': 'Negative',
                    'score': -abs(numerical_score) if numerical_score != 0 else -0.5,
                    'correction': 'direct_symptom_description',
                    'context': 'Patient describing symptoms directly'
                }
            
            # Pattern 4: IMPROVEMENT INDICATORS (should be POSITIVE)
            improvement_words = [
                'better', 'improved', 'improving', 'good', 'fine', 'okay', 'ok',
                'well', 'normal', 'stable', 'manageable', 'tolerable'
            ]
            
            has_improvement = any(word in text_lower for word in improvement_words)
            if has_improvement and not has_negation:  # Don't double-correct
                return {
                    'sentiment': 'Positive',
                    'score': abs(numerical_score) if numerical_score != 0 else 0.7,
                    'correction': 'improvement_indicator',
                    'context': 'Patient indicating improvement'
                }
            
            # Pattern 5: UNCERTAINTY ABOUT SYMPTOMS (should be more NEUTRAL)
            uncertainty_words = [
                'maybe', 'perhaps', 'possibly', 'might', 'could be', 'not sure',
                'think so', 'guess', 'probably', 'sometimes', 'occasionally'
            ]
            
            has_uncertainty = any(phrase in text_lower for phrase in uncertainty_words)
            if has_uncertainty:
                return {
                    'sentiment': 'Neutral',
                    'score': 0.1,  # Slight positive bias for uncertainty about symptoms
                    'correction': 'uncertainty_neutralization',
                    'context': 'Patient expressing uncertainty about symptoms'
                }
        
        # Pattern 6: DOCTOR RESPONSES - Different logic for healthcare providers
        elif speaker == 'DOCTOR':
            # Doctors expressing concern should remain negative
            concern_words = ['concerned', 'worried', 'serious', 'problematic', 'concerning']
            
            if any(word in text_lower for word in concern_words):
                return {
                    'sentiment': 'Negative',
                    'score': -abs(numerical_score) if numerical_score != 0 else -0.4,
                    'correction': 'doctor_concern',
                    'context': 'Doctor expressing medical concern'
                }
            
            # Doctors providing reassurance should be positive
            reassurance_words = ['normal', 'fine', 'good', 'healthy', 'stable', 'improving']
            
            if any(word in text_lower for word in reassurance_words):
                return {
                    'sentiment': 'Positive', 
                    'score': abs(numerical_score) if numerical_score != 0 else 0.6,
                    'correction': 'doctor_reassurance',
                    'context': 'Doctor providing reassurance'
                }
        
        # No correction needed
        return {
            'sentiment': sentiment_label,
            'score': numerical_score,
            'correction': 'none',
            'context': 'No contextual correction applied'
        }
    
    def reset_conversation_history(self):
        """Reset conversation history for new conversation"""
        self.conversation_history = []

class BatchAudioAnalyzer:
    def __init__(self):
        print("Initializing Enhanced Batch Audio Analyzer...")
        print("⚡ Loading all models at startup for immediate availability...")
        
        # ================================
        # IMMEDIATE MODEL LOADING
        # ================================
        self._models_loaded = False
        self._phase_models_loaded = False
        self._phase_models_failed = False
        self._xlm_roberta_loaded = False
        self._bert_loaded = False
        self._keywords_loaded = False
        
        # Basic initialization
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = LabelEncoder()
        self.models = {"lr": None, "svm": None}
        self.model_accuracy = {}
        self.vader = SentimentIntensityAnalyzer()
        
        # Cache files
        self.cached_metrics = None
        self.last_metrics_data_hash = None
        self.metrics_cache_file = os.path.join('models', 'metrics_cache.json')
        
        # Model directories - Using pre-trained models instead of fine-tuned
        # self.distilbert_model_dir = './distilbert-finetuned-patient'  # Replaced with XLM-RoBERTa
        self.phase_model_dir = './distilbert-finetuned-phase'
        
        # NLTK tools initialization
        self.lemmatizer = None
        self.stop_words = None
        self._init_nltk_tools()
        
        # Load all models immediately at startup
        print("🔄 Loading XLM-RoBERTa sentiment model...")
        self._load_xlm_roberta_sentiment()
        
        print("🔄 Initializing BiLSTM contextual sentiment analyzer...")
        self._load_bilstm_contextual_analyzer()
        
        print("🔄 Loading phase classification models...")
        self._load_phase_models()
        
        print("🔄 Loading time keywords...")
        self._load_keywords()
        
        print("🔄 Initializing traditional ML models...")
        self.initialize_models()
        
        print("✅ Enhanced Batch Audio Analyzer fully initialized with all models loaded!")
    
    def _init_nltk_tools(self):
        """Initialize NLTK tools with error handling"""
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            print("NLTK tools initialized successfully")
        except Exception as e:
            print(f"[WARNING] Error initializing NLTK tools: {e}")
            # Fallback to basic stop words
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    def _load_xlm_roberta_sentiment(self):
        """Load XLM-RoBERTa sentiment model (use cached version)"""
        if not self._xlm_roberta_loaded and TRANSFORMERS_AVAILABLE:
            print(f"[OPTIMIZATION] Loading XLM-RoBERTa sentiment model from cache...")
            try:
                print(f"[{DEVICE_NAME}] Loading XLM-RoBERTa sentiment model...")
                
                # First try to use the cached model without downloading
                model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
                print(f"[{DEVICE_NAME}] Using cached model: {model_name}")
                
                try:
                    # Load model and tokenizer separately with local_files_only, then create pipeline
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    
                    # Load tokenizer and model from cache only
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        local_files_only=True,
                        use_auth_token=False
                    )
                    
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        local_files_only=True,
                        use_auth_token=False,
                        torch_dtype=torch.float16 if DEVICE != -1 else torch.float32
                    )
                    
                    # Create pipeline with loaded model and tokenizer
                    self.xlm_roberta_sentiment = hf_pipeline(
                        "sentiment-analysis", 
                        model=model,
                        tokenizer=tokenizer,
                        device=DEVICE,  # Use GPU if available
                        batch_size=16 if DEVICE != -1 else 8
                    )
                    print(f"[{DEVICE_NAME}] Successfully loaded from cache: {model_name}")
                    
                except Exception as cache_error:
                    print(f"[WARNING] Cache loading failed: {cache_error}")
                    print(f"[{DEVICE_NAME}] Attempting to load without local_files_only constraint...")
                    
                    # Fallback: Load normally (may download if needed)
                    self.xlm_roberta_sentiment = hf_pipeline(
                        "sentiment-analysis", 
                        model=model_name,
                        device=DEVICE,
                        use_fast=True,
                        batch_size=16 if DEVICE != -1 else 8,
                        torch_dtype=torch.float16 if DEVICE != -1 else torch.float32
                    )
                    print(f"[{DEVICE_NAME}] Successfully loaded: {model_name}")
                
                if hasattr(self, 'xlm_roberta_sentiment') and self.xlm_roberta_sentiment:
                    print(f"[OPTIMIZATION] XLM-RoBERTa sentiment model loaded successfully")
                else:
                    print("XLM-RoBERTa sentiment model failed to load, falling back to VADER only")
                    self.xlm_roberta_sentiment = None
                    
            except Exception as e:
                print(f"Could not load XLM-RoBERTa sentiment model: {e}")
                self.xlm_roberta_sentiment = None
            self._xlm_roberta_loaded = True
    
    def _load_bilstm_contextual_analyzer(self):
        """Initialize BiLSTM contextual sentiment analyzer"""
        try:
            if hasattr(self, 'xlm_roberta_sentiment') and self.xlm_roberta_sentiment:
                print(f"[OPTIMIZATION] Initializing BiLSTM contextual analyzer with XLM-RoBERTa...")
                self.contextual_analyzer = BiLSTMContextualSentimentAnalyzer(
                    xlm_roberta_pipeline=self.xlm_roberta_sentiment,
                    device=DEVICE if DEVICE != -1 else 'cpu'
                )
                print(f"[OPTIMIZATION] BiLSTM contextual sentiment analyzer initialized successfully")
            else:
                print("[WARNING] XLM-RoBERTa not available, skipping BiLSTM contextual analyzer")
                self.contextual_analyzer = None
        except Exception as e:
            print(f"[WARNING] Could not initialize BiLSTM contextual analyzer: {e}")
            self.contextual_analyzer = None
    
    def _load_phase_models(self):
        """Lazy load phase classification models"""
        if not self._phase_models_loaded and not self._phase_models_failed:
            print(f"[OPTIMIZATION] Loading phase classification models...")
            try:
                self.phase_clf = joblib.load('phase_classifier.pkl')
                self.phase_vectorizer = joblib.load('phase_vectorizer.pkl')
                
                # Skip transformer model loading for now to avoid meta tensor issues
                print(f"[FALLBACK] Using traditional ML models only for phase classification")
                self.phase_model = None
                self.phase_tokenizer = None
                self.phase_label_map = {0: 'before', 1: 'during', 2: 'after'}
                print('[OPTIMIZATION] Traditional phase models loaded successfully')
                self._phase_models_loaded = True
                
            except Exception as e:
                print(f"Error loading phase models: {e}")
                self._phase_models_failed = True
                # Set up fallback for phase classification
                self.phase_model = None
                self.phase_tokenizer = None
                self.phase_clf = None
                self.phase_vectorizer = None
                print("[FALLBACK] Phase classification will use keyword-based fallback")
    
    def _load_keywords(self):
        """Lazy load time keywords"""
        if not self._keywords_loaded:
            self.before_kw, self.during_kw, self.after_kw = self.load_time_keywords_from_csv(
                'C:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\common_synonyms_before_during_after_1000.csv'
            )
            self._keywords_loaded = True

    def initialize_models(self):
        """Initialize or train models"""
        if not self._models_loaded:
            try:
                self.load_models()
                print("[OPTIMIZATION] Traditional ML models loaded successfully")
                self._models_loaded = True
            except Exception as e:
                print(f"Error loading models: {e}")
                self.train_models_with_csv_data()
                self._models_loaded = True
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        os.makedirs('models', exist_ok=True)
        self.models['lr'] = joblib.load('models/lr_model.pkl')
        self.models['svm'] = joblib.load('models/svm_model.pkl')
        self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        self.label_encoder = joblib.load('models/label_encoder.pkl')
    
    def save_models(self):
        """Save models to disk"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.models['lr'], 'models/lr_model.pkl')
        joblib.dump(self.models['svm'], 'models/svm_model.pkl')
        joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
    
    def clean_labels(self, series):
        """Map string labels to numeric and drop invalid/missing labels."""
        label_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1, -1: -1, 0: 0, 1: 1}
        mapped = series.map(label_map)
        # Drop NaNs (invalid labels)
        return mapped.dropna().astype(int)
    
    def load_training_data(self):
        """Load training data from CSV file with statement/label columns, always as DataFrame."""
        if not os.path.exists(TRAINING_DATA_FILE):
            print(f"Training data file '{TRAINING_DATA_FILE}' not found. Using fallback training.")
            return pd.DataFrame({'statement': ['This is good', 'This is bad', 'This is okay'], 'label': [1, -1, 0]})
        try:
            df = pd.read_csv(TRAINING_DATA_FILE)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if 'statement' not in df.columns or 'label' not in df.columns:
                print("CSV must contain 'statement' and 'label' columns. Using fallback training.")
                return pd.DataFrame({'statement': ['This is good', 'This is bad', 'This is okay'], 'label': [1, -1, 0]})
            df = df.dropna(subset=['statement', 'label'])
            df = df[df['statement'].apply(lambda x: isinstance(x, str))]
            # Clean labels
            df['label'] = self.clean_labels(df['label'])
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if len(df) < 10:
                print("Insufficient training data (need at least 10 samples). Using fallback training.")
                return pd.DataFrame({'statement': ['This is good', 'This is bad', 'This is okay'], 'label': [1, -1, 0]})
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"Error loading training data: {e}. Using fallback training.")
            return pd.DataFrame({'statement': ['This is good', 'This is bad', 'This is okay'], 'label': [1, -1, 0]})

    def split_train_val_test(self, X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
        # First, split off the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Now split the remaining data into train and validation
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models_with_csv_data(self):
        """Train models with data from CSV file, with improved feature extraction and class balance check."""
        df = self.load_training_data()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # Preprocess and filter out empty/short/whitespace-only statements
        df['cleaned_text'] = df['statement'].apply(self.preprocess_text)
        df = df[df['cleaned_text'].str.len() > 2]
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # Print class distribution
        class_counts = df['label'].value_counts().sort_index()
        print("Class distribution:")
        for k in [-1, 0, 1]:
            print(f"  {k}: {class_counts.get(k, 0)} samples")
        min_class = class_counts.min() if not class_counts.empty else 0
        max_class = class_counts.max() if not class_counts.empty else 0
        if min_class < 0.2 * max_class:
            print("WARNING: Your data is highly imbalanced. Consider collecting more samples for minority classes.")
        X = df['cleaned_text']
        y = self.label_encoder.fit_transform(df['label'])
        # Improved vectorizer: bigrams, stopwords, sublinear_tf, more features
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)
        X_vec = self.vectorizer.fit_transform(X)
        # Split into train, val, test
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_train_val_test(X_vec, y)
        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
        self.models = {}
        # Logistic Regression
        self.models['lr'] = LogisticRegression(max_iter=1000)
        self.models['lr'].fit(X_train, y_train)
        # SVM
        self.models['svm'] = SVC(kernel='linear', probability=True)
        self.models['svm'].fit(X_train, y_train)
        # Accuracy
        self.model_accuracy = {
            'lr_val': self.models['lr'].score(X_val, y_val),
            'svm_val': self.models['svm'].score(X_val, y_val),
            'lr_test': self.models['lr'].score(X_test, y_test),
            'svm_test': self.models['svm'].score(X_test, y_test)
        }
        self.save_models()
        print(f"Models trained successfully with {len(df)} samples")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"LR Validation Accuracy: {self.model_accuracy['lr_val']:.2f}, SVM Validation Accuracy: {self.model_accuracy['svm_val']:.2f}")
        print(f"LR Test Accuracy: {self.model_accuracy['lr_test']:.2f}, SVM Test Accuracy: {self.model_accuracy['svm_test']:.2f}")
    
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text):
        """
        OPTIMIZATION: Cached and optimized text preprocessing using pre-compiled regex patterns
        """
        text = str(text)
        if not text or text.isspace() or len(text.strip()) < 3:
            return ''
        
        # Use pre-compiled regex patterns
        text = text.lower()
        text = COMPILED_PATTERNS['punctuation_numbers'].sub('', text)
        text = COMPILED_PATTERNS['whitespace'].sub(' ', text).strip()
        
        # Use lazy-loaded NLP tools with error handling
        try:
            tokens = word_tokenize(text)
            stop_words = self.stop_words
            
            tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
            
            # Use lemmatizer with error handling
            if self.lemmatizer:
                try:
                    tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
                except Exception as lemma_error:
                    print(f"[WARNING] Lemmatization error: {lemma_error}")
                    # Continue without lemmatization
                    pass
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"[WARNING] Text preprocessing error: {e}")
            # Fallback to basic preprocessing
            words = text.split()
            basic_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            words = [w for w in words if w not in basic_stop_words and len(w) > 2]
            return ' '.join(words)
    
    def split_statements(self, text):
        """Split input into sentences for future queries"""
        try:
            blob = TextBlob(text)
            sentences_attr = getattr(blob, 'sentences', None)
            # Check if sentences_attr is iterable
            if sentences_attr is not None and hasattr(sentences_attr, '__iter__'):
                sentences = list(sentences_attr)
                if not sentences:
                    return [text]
                return [str(sentence) for sentence in sentences]
            else:
                return [text]
        except Exception as e:
            print(f"TextBlob sentence split error: {e}")
            return [text]

    @cache_result(cache_type='phase')
    def classify_time_period(self, sentence, current_phase=None, debug=False):
        """
        Enhanced phase classification with multi-layered approach and improved accuracy
        """
        # Phase models are already loaded at startup
        
        # If traditional ML phase models are available, use them first
        if self._phase_models_loaded and self.phase_clf and self.phase_vectorizer:
            try:
                # Use traditional ML for phase classification
                cleaned_sentence = self.preprocess_text(sentence)
                if cleaned_sentence:
                    vectorized = self.phase_vectorizer.transform([cleaned_sentence])
                    pred = self.phase_clf.predict(vectorized)[0]
                    ml_confidence = max(self.phase_clf.predict_proba(vectorized)[0])
                    ml_phase = self.phase_label_map.get(pred, 'during')
                    
                    # If ML model is confident enough, use its prediction
                    if ml_confidence > 0.7:
                        if debug:
                            print(f"[PHASE ML] '{sentence}' => {ml_phase} (confidence: {ml_confidence:.3f})")
                        return ml_phase, ml_phase
                    else:
                        # Use enhanced fallback for low confidence predictions
                        enhanced_phase = self._enhanced_phase_classification(sentence, debug)
                        if debug:
                            print(f"[PHASE ML+ENHANCED] '{sentence}' => ML: {ml_phase} ({ml_confidence:.3f}), Enhanced: {enhanced_phase}")
                        return enhanced_phase, enhanced_phase
            except Exception as e:
                print(f"[WARNING] Traditional ML phase classification error: {e}")
        
        # Enhanced fallback classification
        enhanced_phase = self._enhanced_phase_classification(sentence, debug)
        if debug:
            print(f"[PHASE ENHANCED] '{sentence}' => {enhanced_phase}")
        return enhanced_phase, enhanced_phase
    
    def _enhanced_phase_classification(self, sentence, debug=False):
        """
        Enhanced multi-layered phase classification using comprehensive keyword matching,
        temporal patterns, grammatical analysis, and contextual clues
        """
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        
        # Initialize scores
        before_score = 0
        during_score = 0
        after_score = 0
        
        # Layer 1: Enhanced keyword matching with weights
        # Keywords are already loaded at startup
        
        # Comprehensive keyword sets with weights
        before_keywords = {
            # Time indicators (high weight)
            'before': 3, 'prior': 3, 'previously': 3, 'earlier': 3, 'past': 2,
            'initially': 2, 'originally': 2, 'formerly': 2, 'beforehand': 3,
            # Past tense indicators (medium weight) 
            'was': 1, 'were': 1, 'had': 1, 'used to': 2, 'would': 1,
            # Medical history indicators (high weight)
            'history': 2, 'diagnosed': 2, 'started': 2, 'began': 2, 'onset': 3,
            'first': 2, 'initial': 2, 'baseline': 2, 'pre-treatment': 3,
            # Condition descriptors (medium weight)
            'chronic': 1, 'persistent': 1, 'long-standing': 2, 'deteriorating': 2,
            'worsening': 2, 'progressive': 2, 'advanced': 1
        }
        
        during_keywords = {
            # Current time indicators (high weight)
            'during': 3, 'while': 2, 'currently': 3, 'now': 2, 'present': 2,
            'today': 2, 'this': 1, 'experiencing': 2, 'ongoing': 2,
            # Treatment indicators (high weight)
            'treatment': 2, 'therapy': 2, 'medication': 2, 'procedure': 3,
            'surgery': 3, 'operation': 3, 'session': 2, 'appointment': 2,
            # Current state indicators (medium weight)
            'feeling': 1, 'having': 1, 'getting': 1, 'receiving': 2,
            'undergoing': 3, 'managing': 2, 'coping': 2, 'dealing': 1,
            # Process indicators (medium weight)
            'adjusting': 2, 'monitoring': 2, 'tracking': 2, 'working': 1
        }
        
        after_keywords = {
            # Future/completion indicators (high weight)
            'after': 3, 'following': 3, 'since': 2, 'future': 2, 'will': 1,
            'next': 1, 'plan': 2, 'hope': 1, 'expect': 1, 'post': 3,
            # Recovery indicators (high weight)
            'recovered': 3, 'better': 2, 'improved': 3, 'healed': 3,
            'cured': 3, 'resolved': 3, 'successful': 2, 'effective': 2,
            # Outcome indicators (medium weight)
            'result': 1, 'outcome': 2, 'follow-up': 3, 'discharge': 3,
            'complete': 2, 'finished': 2, 'done': 1, 'over': 1,
            # Future planning (medium weight)
            'maintenance': 2, 'prevention': 2, 'monitor': 1, 'check': 1
        }
        
        # Calculate keyword scores
        for word in words:
            before_score += before_keywords.get(word, 0)
            during_score += during_keywords.get(word, 0)
            after_score += after_keywords.get(word, 0)
        
        # Layer 2: Grammatical tense analysis
        tense_score = self._analyze_grammatical_tense(sentence_lower)
        before_score += tense_score['past']
        during_score += tense_score['present']
        after_score += tense_score['future']
        
        # Layer 3: Contextual pattern matching
        context_score = self._analyze_contextual_patterns(sentence_lower)
        before_score += context_score['before']
        during_score += context_score['during']
        after_score += context_score['after']
        
        # Layer 4: Medical domain-specific indicators
        medical_score = self._analyze_medical_indicators(sentence_lower)
        before_score += medical_score['before']
        during_score += medical_score['during']
        after_score += medical_score['after']
        
        # Layer 5: Load expanded keywords from CSV if available
        csv_score = self._analyze_csv_keywords(sentence_lower)
        before_score += csv_score['before']
        during_score += csv_score['during']
        after_score += csv_score['after']
        
        # Determine final phase with confidence threshold
        max_score = max(before_score, during_score, after_score)
        
        # Reduce minimum confidence gap to make classification more sensitive
        min_confidence_gap = 0.5  # Reduced from 1 to 0.5
        
        if max_score == 0:
            # No indicators found, default to 'during'
            return 'during'
        elif before_score == max_score and before_score - max(during_score, after_score) >= min_confidence_gap:
            return 'before'
        elif after_score == max_score and after_score - max(before_score, during_score) >= min_confidence_gap:
            return 'after'
        else:
            # Default to 'during' if scores are too close or during wins
            return 'during'
    
    def _analyze_grammatical_tense(self, sentence):
        """Analyze grammatical tense indicators"""
        scores = {'past': 0, 'present': 0, 'future': 0}
        
        # Past tense patterns
        past_patterns = [
            r'\b(was|were|had|did|been)\b',
            r'\b\w+ed\b',  # Past tense verbs ending in -ed
            r'\b(ago|yesterday|last)\b'
        ]
        
        # Present tense patterns  
        present_patterns = [
            r'\b(am|is|are|have|has|do|does)\b',
            r'\b(currently|now|today|presently)\b',
            r'\b\w+ing\b'  # Present participle verbs
        ]
        
        # Future tense patterns
        future_patterns = [
            r'\b(will|shall|going to|would|could|should)\b',
            r'\b(tomorrow|next|future|plan|hope)\b'
        ]
        
        for pattern in past_patterns:
            scores['past'] += len(re.findall(pattern, sentence, re.IGNORECASE))
        
        for pattern in present_patterns:
            scores['present'] += len(re.findall(pattern, sentence, re.IGNORECASE))
            
        for pattern in future_patterns:
            scores['future'] += len(re.findall(pattern, sentence, re.IGNORECASE))
        
        return scores
    
    def _analyze_contextual_patterns(self, sentence):
        """Analyze contextual patterns and phrases"""
        scores = {'before': 0, 'during': 0, 'after': 0}
        
        # Multi-word phrases that are strong indicators
        before_phrases = [
            'before treatment', 'prior to', 'in the past', 'used to be',
            'initial symptoms', 'first time', 'when it started', 'originally',
            'back then', 'at first', 'early on', 'months ago', 'years ago'
        ]
        
        during_phrases = [
            'right now', 'at the moment', 'these days', 'currently experiencing',
            'in treatment', 'on medication', 'during therapy', 'while taking',
            'at present', 'this week', 'this month', 'as we speak'
        ]
        
        after_phrases = [
            'after treatment', 'post surgery', 'following therapy', 'once healed',
            'in the future', 'moving forward', 'from now on', 'going forward',
            'recovery phase', 'follow up', 'next steps', 'long term'
        ]
        
        # Check for phrase matches
        for phrase in before_phrases:
            if phrase in sentence:
                scores['before'] += 2
        
        for phrase in during_phrases:
            if phrase in sentence:
                scores['during'] += 2
                
        for phrase in after_phrases:
            if phrase in sentence:
                scores['after'] += 2
        
        return scores
    
    def _analyze_medical_indicators(self, sentence):
        """Analyze medical domain-specific temporal indicators"""
        scores = {'before': 0, 'during': 0, 'after': 0}
        
        # Medical before indicators
        if any(word in sentence for word in ['diagnosis', 'diagnosed', 'symptoms started', 'onset', 'first noticed']):
            scores['before'] += 2
        
        # Medical during indicators  
        if any(word in sentence for word in ['treatment', 'therapy', 'medication', 'procedure', 'surgery', 'taking']):
            scores['during'] += 2
            
        # Medical after indicators
        if any(word in sentence for word in ['recovery', 'healed', 'discharged', 'follow-up', 'maintenance']):
            scores['after'] += 2
        
        return scores
    
    def _analyze_csv_keywords(self, sentence):
        """Analyze keywords from CSV files"""
        scores = {'before': 0, 'during': 0, 'after': 0}
        
        try:
            # Load keywords from CSV if available
            if hasattr(self, 'before_keywords_csv') and hasattr(self, 'during_keywords_csv') and hasattr(self, 'after_keywords_csv'):
                words = sentence.split()
                
                for word in words:
                    if word in self.before_keywords_csv:
                        scores['before'] += 1
                    if word in self.during_keywords_csv:
                        scores['during'] += 1
                    if word in self.after_keywords_csv:
                        scores['after'] += 1
            else:
                # Load keywords from CSV files
                self._load_csv_keywords()
                # Recursive call after loading
                return self._analyze_csv_keywords(sentence)
                
        except Exception as e:
            # Silent fail for CSV keyword analysis
            pass
        
        return scores
    
    def _load_csv_keywords(self):
        """Load keywords from CSV files"""
        try:
            # Load from the comprehensive synonyms file
            csv_path = "c:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\common_synonyms_before_during_after_1000.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                self.before_keywords_csv = set()
                self.during_keywords_csv = set()
                self.after_keywords_csv = set()
                
                if 'Before' in df.columns:
                    self.before_keywords_csv = set(str(x).strip().lower() for x in df['Before'].dropna() if str(x).strip())
                if 'During' in df.columns:
                    self.during_keywords_csv = set(str(x).strip().lower() for x in df['During'].dropna() if str(x).strip())
                if 'After' in df.columns:
                    self.after_keywords_csv = set(str(x).strip().lower() for x in df['After'].dropna() if str(x).strip())
                    
            # Also load from the sentiment phases file
            phases_path = "c:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\sentiment_phases_1000.csv"
            if os.path.exists(phases_path):
                df2 = pd.read_csv(phases_path)
                
                if 'Before' in df2.columns:
                    self.before_keywords_csv.update(str(x).strip().lower() for x in df2['Before'].dropna() if str(x).strip())
                if 'During' in df2.columns:
                    self.during_keywords_csv.update(str(x).strip().lower() for x in df2['During'].dropna() if str(x).strip())
                if 'After' in df2.columns:
                    self.after_keywords_csv.update(str(x).strip().lower() for x in df2['After'].dropna() if str(x).strip())
                    
        except Exception as e:
            print(f"Error loading CSV keywords: {e}")
            # Set empty sets as fallback
            self.before_keywords_csv = set()
            self.during_keywords_csv = set()
            self.after_keywords_csv = set()
    
    def predict_phase(self, text, debug=False):
        """Convenient method for phase prediction with detailed output"""
        phase, _ = self.classify_time_period(text, debug=debug)
        confidence = self._calculate_phase_confidence(text, phase)
        
        return {
            'phase': phase,
            'confidence': confidence,
            'text': text
        }
    
    def _calculate_phase_confidence(self, text, predicted_phase):
        """Calculate confidence score for the predicted phase"""
        try:
            # Get scores from enhanced classification
            sentence_lower = text.lower()
            words = sentence_lower.split()
            
            before_score = 0
            during_score = 0
            after_score = 0
            
            # Simplified scoring for confidence calculation
            before_keywords = ['before', 'prior', 'previously', 'earlier', 'past', 'was', 'were', 'initially', 'originally']
            during_keywords = ['during', 'while', 'currently', 'now', 'present', 'today', 'experiencing', 'treatment', 'therapy']
            after_keywords = ['after', 'following', 'since', 'future', 'will', 'next', 'plan', 'hope', 'recovered', 'better']
            
            for word in words:
                if word in before_keywords:
                    before_score += 1
                if word in during_keywords:
                    during_score += 1
                if word in after_keywords:
                    after_score += 1
            
            total_score = before_score + during_score + after_score
            if total_score == 0:
                return 0.5  # Neutral confidence when no indicators
            
            # Calculate confidence based on the predicted phase score
            if predicted_phase == 'before':
                confidence = before_score / total_score
            elif predicted_phase == 'during':
                confidence = during_score / total_score
            elif predicted_phase == 'after':
                confidence = after_score / total_score
            else:
                confidence = 0.5
            
            return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
            
        except Exception as e:
            return 0.5  # Default confidence if calculation fails

    def calculate_patient_satisfaction_score(self, patient_sentiments, debug=False):
        """
        Calculate patient satisfaction score using Weighted Average Aggregation with Dampened Negative Impact
        
        Mathematical Formula:
        1. Dampening Negative Scores: s'_i = s_i if s_i >= 0, else α × s_i where 0 < α < 1
        2. Weighted Average: SI = Σ(w_i × s'_i) / Σ(w_i)
        3. Optional Normalization: SI_norm = 100 × tanh(k × SI)
        """
        if not patient_sentiments:
            return {
                'satisfaction_score': 50.0,  # Neutral baseline
                'satisfaction_level': "Neutral",
                'level_emoji': "",
                'calculation_details': {
                    'raw_score': 0,
                    'dampened_scores': [],
                    'weights': [],
                    'final_normalized': 50.0
                }
            }
        
        # ================================
        # STEP 1: Extract sentiment scores and prepare weights
        # ================================
        raw_scores = []
        weights = []
        
        for sentiment_data in patient_sentiments:
            # Extract numerical score (assuming -3 to +3 scale)
            if 'numerical_score' in sentiment_data:
                score = sentiment_data['numerical_score']
            elif 'score' in sentiment_data:
                # Convert sentiment score to numerical if needed
                score = float(sentiment_data['score'])
                # Normalize to -3 to +3 scale if needed
                if abs(score) > 3:
                    score = score / abs(score) * min(3, abs(score) / 100 * 3)
            else:
                # Fallback based on sentiment label
                sentiment = sentiment_data.get('sentiment', 'Neutral')
                if sentiment == 'Positive':
                    score = 1.5
                elif sentiment == 'Negative':
                    score = -1.5
                else:
                    score = 0
            
            raw_scores.append(score)
            
            # Assign weights based on confidence, phase importance, etc.
            weight = 1.0  # Default uniform weight
            
            # Weight by phase importance (clinical relevance)
            phase = sentiment_data.get('phase', 'during').lower()
            if 'before' in phase:
                weight *= 0.8  # Baseline expectations
            elif 'during' in phase:
                weight *= 1.2  # Treatment experience (most important)
            elif 'after' in phase:
                weight *= 1.0  # Outcome satisfaction
            
            # Weight by confidence/model quality if available
            if 'confidence' in sentiment_data:
                confidence = sentiment_data['confidence']
                weight *= (0.5 + 0.5 * confidence)  # Scale weight by confidence
            
            weights.append(weight)
        
        # ================================
        # STEP 2: Apply Dampening Function to Negative Scores
        # ================================
        alpha = 0.5  # Dampening factor for negative sentiments (0 < α < 1)
        dampened_scores = []
        
        for score in raw_scores:
            if score >= 0:
                dampened_score = score  # Keep positive scores unchanged
            else:
                dampened_score = alpha * score  # Dampen negative scores
            dampened_scores.append(dampened_score)
        
        # ================================
        # STEP 3: Weighted Average Aggregation
        # ================================
        if sum(weights) > 0:
            # Calculate weighted average: SI = Σ(w_i × s'_i) / Σ(w_i)
            weighted_sum = sum(w * s for w, s in zip(weights, dampened_scores))
            total_weight = sum(weights)
            satisfaction_index_raw = weighted_sum / total_weight
        else:
            satisfaction_index_raw = 0
        
        # ================================
        # STEP 4: Optional Normalization & Scaling
        # ================================
        k = 1.0  # Scaling constant for tanh normalization
        
        # Apply tanh normalization: SI_norm = 100 × tanh(k × SI)
        # This constrains the result to roughly [-100, 100] range
        satisfaction_index_tanh = 100 * np.tanh(k * satisfaction_index_raw)
        
        # Convert to 0-100 scale for interpretability
        # Map from [-100, 100] to [0, 100]
        satisfaction_score = 50 + (satisfaction_index_tanh / 2)
        
        # Ensure score stays within 0-100 range
        satisfaction_score = max(0, min(100, satisfaction_score))
        
        # ================================
        # STEP 5: Satisfaction Level Classification
        # ================================
        if satisfaction_score >= 80:
            satisfaction_level = "Excellent"
            level_emoji = "😊"
        elif satisfaction_score >= 65:
            satisfaction_level = "Good"
            level_emoji = "🙂"
        elif satisfaction_score >= 45:
            satisfaction_level = "Satisfactory"
            level_emoji = "😐"
        elif satisfaction_score >= 30:
            satisfaction_level = "Poor"
            level_emoji = "😞"
        else:
            satisfaction_level = "Very Poor"
            level_emoji = "😡"
            
        
        # ================================
        # DEBUG OUTPUT (if requested)
        # ================================
        if debug:
            print(f"[WEIGHTED AVERAGE AGGREGATION WITH DAMPENED NEGATIVE IMPACT]")
            print(f"  Raw scores: {[f'{s:.2f}' for s in raw_scores]}")
            print(f"  Weights: {[f'{w:.2f}' for w in weights]}")
            print(f"  Dampened scores (α={alpha}): {[f'{s:.2f}' for s in dampened_scores]}")
            print(f"  Weighted sum: {weighted_sum:.3f}")
            print(f"  Total weight: {total_weight:.3f}")
            print(f"  Raw weighted average: {satisfaction_index_raw:.3f}")
            print(f"  Tanh normalized: {satisfaction_index_tanh:.3f}")
            print(f"  Final score (0-100): {satisfaction_score:.1f}")
            print(f"  Classification: {satisfaction_level}")
        
        return {
            'satisfaction_score': round(satisfaction_score, 1),
            'satisfaction_level': satisfaction_level,
            'level_emoji': level_emoji,
            'calculation_details': {
                'raw_scores': [round(s, 3) for s in raw_scores],
                'dampened_scores': [round(s, 3) for s in dampened_scores],
                'weights': [round(w, 3) for w in weights],
                'alpha_dampening_factor': alpha,
                'raw_weighted_average': round(satisfaction_index_raw, 3),
                'tanh_normalized': round(satisfaction_index_tanh, 3),
                'final_normalized': round(satisfaction_score, 1),
                'scaling_constant_k': k
            }
        }

    def analyze_sentiment_trends(self, patient_sentiments):
        """
        Analyze temporal sentiment trends and patterns
        """
        if len(patient_sentiments) < 2:
            return {
                'trend_description': "Insufficient data for trend analysis",
                'volatility_description': "Insufficient data for volatility analysis",
                'trend_direction': 0,
                'volatility_score': 0
            }
        
        # Extract numerical scores in chronological order (approximated by phase)
        phase_order = {'before': 1, 'during': 2, 'after': 3}
        
        # Sort sentiments by phase order
        sorted_sentiments = sorted(patient_sentiments, 
                                 key=lambda x: phase_order.get(x.get('phase', 'during').lower(), 2))
        
        scores = [s.get('numerical_score', 0) for s in sorted_sentiments]
        
        # Calculate trend direction
        if len(scores) >= 2:
            start_score = np.mean(scores[:len(scores)//2])
            end_score = np.mean(scores[len(scores)//2:])
            trend_direction = end_score - start_score
            
            if trend_direction > 0.5:
                trend_description = "Improving sentiment over time 📈"
            elif trend_direction < -0.5:
                trend_description = "Declining sentiment over time 📉"
            else:
                trend_description = "Stable sentiment throughout visit ➡️"
        else:
            trend_description = "Limited trend data available"
            trend_direction = 0
        
        # Identify sentiment volatility
        if len(scores) > 2:
            volatility = np.std(scores)
            if volatility > 1.5:
                volatility_desc = "High sentiment variability (mixed experience)"
            elif volatility < 0.5:
                volatility_desc = "Low sentiment variability (consistent experience)"
            else:
                volatility_desc = "Moderate sentiment variability"
        else:
            volatility_desc = "Insufficient data for volatility analysis"
            volatility = 0
        
        return {
            'trend_description': trend_description,
            'volatility_description': volatility_desc,
            'trend_direction': trend_direction,
            'volatility_score': volatility
        }

    def generate_conversation_summary(self, patient_sentiments, all_sentiments, satisfaction_results):
        """
        Generate comprehensive conversation summary with quality metrics
        """
        try:
            # Basic statistics
            total_segments = len(all_sentiments)
            patient_segments = len(patient_sentiments)
            doctor_segments = total_segments - patient_segments
            
            # Sentiment distribution
            sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
            phase_counts = {'Before': 0, 'During': 0, 'After': 0}
            
            for sentiment in patient_sentiments:
                sentiment_counts[sentiment.get('sentiment', 'Neutral')] += 1
                phase_counts[sentiment.get('phase', 'During')] += 1
            
            # Calculate conversation quality metrics
            avg_satisfaction = satisfaction_results.get('satisfaction_score', 50)
            
            # Conversation balance
            if doctor_segments > 0:
                patient_talk_ratio = patient_segments / total_segments
                conversation_balance = "Balanced" if 0.4 <= patient_talk_ratio <= 0.6 else "Patient-heavy" if patient_talk_ratio > 0.6 else "Doctor-heavy"
            else:
                conversation_balance = "Patient-only"
            
            # Overall sentiment tone
            positive_ratio = sentiment_counts['Positive'] / patient_segments if patient_segments > 0 else 0
            negative_ratio = sentiment_counts['Negative'] / patient_segments if patient_segments > 0 else 0
            
            if positive_ratio > 0.6:
                overall_tone = "Predominantly positive"
            elif negative_ratio > 0.6:
                overall_tone = "Predominantly negative"
            elif positive_ratio > negative_ratio:
                overall_tone = "Moderately positive"
            elif negative_ratio > positive_ratio:
                overall_tone = "Moderately negative"
            else:
                overall_tone = "Mixed/Neutral"
            
            # Key insights
            insights = []
            
            if avg_satisfaction >= 75:
                insights.append("High patient satisfaction indicates positive care experience")
            elif avg_satisfaction <= 35:
                insights.append("Low satisfaction scores suggest areas for care improvement")
            
            if phase_counts['Before'] > phase_counts['After']:
                insights.append("More focus on patient history than follow-up planning")
            elif phase_counts['After'] > phase_counts['Before']:
                insights.append("Strong emphasis on future care and recovery planning")
            
            if negative_ratio > 0.5:
                insights.append("Significant patient concerns identified requiring attention")
            
            if conversation_balance == "Doctor-heavy":
                insights.append("Doctor-led conversation style - consider more patient engagement")
            elif conversation_balance == "Patient-heavy":
                insights.append("Patient-centered discussion with active patient participation")
            
            return {
                'conversation_stats': {
                    'total_segments': total_segments,
                    'patient_segments': patient_segments,
                    'doctor_segments': doctor_segments,
                    'conversation_balance': conversation_balance,
                    'patient_talk_ratio': round(patient_talk_ratio if doctor_segments > 0 else 1.0, 2)
                },
                'sentiment_distribution': sentiment_counts,
                'phase_distribution': phase_counts,
                'overall_assessment': {
                    'satisfaction_score': avg_satisfaction,
                    'overall_tone': overall_tone,
                    'positive_ratio': round(positive_ratio, 2),
                    'negative_ratio': round(negative_ratio, 2)
                },
                'key_insights': insights,
                'summary_text': f"Conversation analysis: {total_segments} total segments with {overall_tone.lower()} patient sentiment. "
                              f"Patient satisfaction: {avg_satisfaction:.1f}/100. {conversation_balance} conversation style. "
                              f"{'Key concerns identified.' if negative_ratio > 0.5 else 'Generally positive interaction.'}"
            }
            
        except Exception as e:
            print(f"[WARNING] Error generating conversation summary: {e}")
            return {
                'error': str(e),
                'summary_text': "Unable to generate comprehensive conversation summary due to processing error."
            }

    def load_time_keywords_from_csv(self, csv_path):
        """Load time-based keywords from CSV file"""
        before_kw, during_kw, after_kw = [], [], []
        try:
            df = pd.read_csv(csv_path)
            if 'before' in df.columns:
                before_kw = [str(x).strip().lower() for x in df['before'].dropna() if str(x).strip()]
            if 'during' in df.columns:
                during_kw = [str(x).strip().lower() for x in df['during'].dropna() if str(x).strip()]
            if 'after' in df.columns:
                after_kw = [str(x).strip().lower() for x in df['after'].dropna() if str(x).strip()]
        except Exception as e:
            print(f"Error loading time keywords from {csv_path}: {e}")
        return before_kw, during_kw, after_kw
    
    def classify_speaker_with_distilbert(self, text):
        """
        Classify speaker (Doctor vs Patient) using fine-tuned DistilBERT model
        """
        try:
            classifier = model_manager.get_speaker_classifier()
            if classifier is None:
                # Fallback to simple heuristic
                return self._fallback_speaker_classification(text)
            
            # Use the fine-tuned model to classify
            results = classifier(text)
            
            # The model outputs scores for each class (0=Doctor, 1=Patient)
            doctor_score = None
            patient_score = None
            
            for result in results[0]:  # results is a list with one item containing all scores
                if result['label'] == 'LABEL_0':  # Doctor
                    doctor_score = result['score']
                elif result['label'] == 'LABEL_1':  # Patient
                    patient_score = result['score']
            
            # Determine the speaker based on highest confidence
            if doctor_score is not None and patient_score is not None:
                if doctor_score > patient_score:
                    speaker = 'DOCTOR'
                    confidence = doctor_score
                else:
                    speaker = 'PATIENT'
                    confidence = patient_score
            else:
                # Fallback if labels are unexpected
                return self._fallback_speaker_classification(text)
            
            return {
                'speaker': speaker,
                'confidence': confidence,
                'doctor_score': doctor_score,
                'patient_score': patient_score,
                'text': text[:50] + "..." if len(text) > 50 else text
            }
            
        except Exception as e:
            print(f"[WARNING] Speaker classification error: {e}")
            return self._fallback_speaker_classification(text)
    
    def _fallback_speaker_classification(self, text):
        """
        Simple fallback speaker classification when DistilBERT model is not available
        """
        text_lower = text.lower()
        
        # Simple keyword-based fallback
        doctor_keywords = [
            'how are you', 'let me', 'i need to', 'examination', 'diagnosis',
            'what brings you', 'when did', 'any pain', 'take this medication',
            'follow up', 'prescription', 'symptoms', 'medical history'
        ]
        
        patient_keywords = [
            'i feel', 'i have', 'my pain', 'it hurts', 'i\'ve been',
            'started', 'since', 'i think', 'i\'m worried', 'help me'
        ]
        
        doctor_score = sum(1 for keyword in doctor_keywords if keyword in text_lower)
        patient_score = sum(1 for keyword in patient_keywords if keyword in text_lower)
        
        if doctor_score > patient_score:
            speaker = 'DOCTOR'
            confidence = 0.6  # Lower confidence for fallback
        elif patient_score > doctor_score:
            speaker = 'PATIENT'
            confidence = 0.6
        else:
            # Default to patient if unclear
            speaker = 'PATIENT'
            confidence = 0.5
        
        return {
            'speaker': speaker,
            'confidence': confidence,
            'doctor_score': doctor_score / len(doctor_keywords) if doctor_keywords else 0,
            'patient_score': patient_score / len(patient_keywords) if patient_keywords else 0,
            'text': text[:50] + "..." if len(text) > 50 else text,
            'method': 'fallback'
        }

    def transcribe_audio(self, audio_file_path):
        """
        Transcribe audio using Whisper model
        """
        try:
            print(f"📝 Transcribing audio: {audio_file_path}")
            
            # Check if file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Get absolute path
            abs_path = os.path.abspath(audio_file_path)
            print(f"🔍 Full path: {abs_path}")
            
            # Check file size
            file_size = os.path.getsize(abs_path)
            print(f"📊 File size: {file_size / (1024*1024):.1f} MB")
            
            # Use Whisper to transcribe
            whisper_model = model_manager.get_whisper_model()
            if whisper_model is None:
                raise Exception("Whisper model not available")
            
            print("🔄 Starting Whisper transcription...")
            
            # Try different approaches for transcription
            try:
                # First, try with minimal parameters
                result = whisper_model.transcribe(abs_path, verbose=False)
                print("✅ Whisper transcription successful with minimal parameters")
            except Exception as e1:
                print(f"⚠️ First attempt failed: {e1}")
                try:
                    # Try with explicit audio loading using librosa if available
                    import librosa
                    audio_data, sr = librosa.load(abs_path, sr=16000)
                    result = whisper_model.transcribe(audio_data, verbose=False)
                    print("✅ Whisper transcription successful with librosa")
                except ImportError:
                    print("📦 librosa not available, trying alternative approach...")
                    # Try with numpy array approach
                    try:
                        import numpy as np
                        from pydub import AudioSegment
                        
                        # Load with pydub and convert to numpy
                        audio = AudioSegment.from_file(abs_path)
                        audio = audio.set_channels(1).set_frame_rate(16000)
                        audio_array = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
                        
                        result = whisper_model.transcribe(audio_array, verbose=False)
                        print("✅ Whisper transcription successful with pydub conversion")
                    except Exception as e3:
                        print(f"❌ All transcription methods failed. Last error: {e3}")
                        raise e3
                except Exception as e2:
                    print(f"❌ Librosa method failed: {e2}")
                    raise e2
            
            # Extract text and metadata
            transcript_text = result.get('text', '').strip()
            language = result.get('language', 'unknown')
            
            if not transcript_text:
                raise Exception("No transcription text generated")
            
            print(f"✅ Transcription completed ({len(transcript_text)} characters, language: {language})")
            print(f"📝 Preview: {transcript_text[:100]}...")
            
            return {
                'text': transcript_text,
                'language': language,
                'segments': result.get('segments', []),
                'duration': result.get('duration', 0),
                'model_used': 'whisper'
            }
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            print("📋 Full traceback:")
            traceback.print_exc()
            return {'text': '', 'error': str(e)}

    def perform_speaker_diarization(self, audio_file_path):
        """
        Perform speaker diarization using pyannote.audio and calculate DER
        """
        try:
            if not PYANNOTE_AVAILABLE:
                print("⚠️ pyannote.audio not available, skipping diarization")
                return None
            
            print("🎭 Performing speaker diarization...")
            
            # Get diarization pipeline
            diarization_pipeline = model_manager.get_diarization_pipeline()
            if diarization_pipeline is None:
                print("❌ Diarization pipeline not available")
                return None
            
            # Perform diarization
            diarization = diarization_pipeline(audio_file_path)
            
            # Extract speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })
            
            print(f"✅ Diarization completed: {len(speaker_segments)} speaker segments")
            
            # Calculate basic diarization metrics
            total_duration = max([seg['end'] for seg in speaker_segments]) if speaker_segments else 0
            unique_speakers = len(set([seg['speaker'] for seg in speaker_segments]))
            
            # Calculate DER (simplified version)
            # For proper DER calculation, we would need ground truth annotations
            # Here we'll calculate a basic quality metric based on speaker consistency
            der_estimate = self.estimate_der(speaker_segments, total_duration)
            
            return {
                'segments': speaker_segments,
                'unique_speakers': unique_speakers,
                'total_duration': total_duration,
                'der_estimate': der_estimate,
                'diarization_object': diarization
            }
            
        except Exception as e:
            print(f"❌ Diarization error: {e}")
            return None

    def estimate_der(self, speaker_segments, total_duration):
        """
        Estimate DER based on speaker consistency and segment quality
        """
        try:
            if not speaker_segments or total_duration == 0:
                return 1.0  # Maximum error
            
            # Calculate segment consistency
            avg_segment_duration = sum([seg['duration'] for seg in speaker_segments]) / len(speaker_segments)
            
            # Penalize very short segments (likely errors)
            short_segments = [seg for seg in speaker_segments if seg['duration'] < 0.5]
            short_segment_penalty = len(short_segments) / len(speaker_segments) * 0.3
            
            # Reward longer, consistent segments
            long_segments = [seg for seg in speaker_segments if seg['duration'] > 2.0]
            long_segment_reward = len(long_segments) / len(speaker_segments) * 0.2
            
            # Calculate coverage
            total_segmented_time = sum([seg['duration'] for seg in speaker_segments])
            coverage = total_segmented_time / total_duration
            
            # Estimate DER (lower is better, 0.0 = perfect, 1.0 = worst)
            base_error = 0.15  # Baseline error
            der_estimate = base_error + short_segment_penalty - long_segment_reward + (1.0 - coverage) * 0.2
            
            # Clamp between 0 and 1
            der_estimate = max(0.0, min(1.0, der_estimate))
            
            return der_estimate
            
        except Exception as e:
            print(f"❌ DER estimation error: {e}")
            return 0.5  # Neutral estimate

    def segment_transcript(self, transcript_text):
        """
        Segment transcript into meaningful chunks for analysis
        """
        try:
            if not transcript_text:
                return []
            
            # Split by sentences using NLTK sentence tokenizer
            try:
                sentences = sent_tokenize(transcript_text)
            except:
                # Fallback: simple sentence splitting
                sentences = [s.strip() for s in transcript_text.split('.') if s.strip()]
            
            # Clean and filter segments
            segments = []
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Skip very short segments (less than 10 characters)
                if len(sentence) < 10:
                    continue
                
                # Skip segments that are mostly punctuation
                alpha_chars = sum(c.isalpha() for c in sentence)
                if alpha_chars < len(sentence) * 0.5:
                    continue
                
                segments.append(sentence)
            
            # If segments are too long, split them further
            final_segments = []
            max_segment_length = 300  # characters
            
            for segment in segments:
                if len(segment) <= max_segment_length:
                    final_segments.append(segment)
                else:
                    # Split long segments by commas or semicolons
                    sub_segments = []
                    for delimiter in [';', ',']:
                        if delimiter in segment:
                            sub_segments = [s.strip() for s in segment.split(delimiter) if s.strip()]
                            break
                    
                    if not sub_segments:
                        # Split by words if no punctuation available
                        words = segment.split()
                        chunk_size = 30  # words per chunk
                        for i in range(0, len(words), chunk_size):
                            chunk = ' '.join(words[i:i + chunk_size])
                            if len(chunk.strip()) >= 10:
                                sub_segments.append(chunk.strip())
                    
                    final_segments.extend(sub_segments)
            
            print(f"📊 Segmented transcript into {len(final_segments)} segments")
            return final_segments
            
        except Exception as e:
            print(f"❌ Segmentation error: {e}")
            # Fallback: return simple split
            return [s.strip() for s in transcript_text.split('.') if s.strip() and len(s.strip()) > 10]

    def analyze_sentiment(self, text, speaker='PATIENT', context=None):
        """
        Analyze sentiment of text using BiLSTM contextual analyzer with XLM-RoBERTa and bias detection
        """
        try:
            if not text or len(text.strip()) < 3:
                return {
                    'sentiment': 'Neutral',
                    'score': 0.0,
                    'numerical_score': 0.0,
                    'confidence': 0.5,
                    'phase': 'during',
                    'method': 'fallback',
                    'bias_analysis': {'potential_bias_detected': False}
                }
            
            # Use BiLSTM contextual analyzer if available
            if hasattr(self, 'contextual_analyzer') and self.contextual_analyzer:
                try:
                    contextual_result = self.contextual_analyzer.analyze_contextual_sentiment(text, speaker)
                    
                    # Add phase classification
                    phase, _ = self.classify_time_period(text)
                    contextual_result['phase'] = phase
                    
                    # Log bias detection results if present
                    if 'bias_analysis' in contextual_result and contextual_result['bias_analysis']['potential_bias_detected']:
                        bias_types = ', '.join(contextual_result['bias_analysis']['bias_types'])
                        print(f"[BIAS DETECTED] {bias_types} | Severity: {contextual_result['bias_analysis']['bias_severity']:.2f}")
                    
                    print(f"[CONTEXTUAL] '{text[:50]}...' => {contextual_result['sentiment']} | Score: {contextual_result['numerical_score']:.2f} | Method: {contextual_result['method']}")
                    
                    return contextual_result
                    
                except Exception as e:
                    print(f"[WARNING] Contextual sentiment analysis failed: {e}")
                    # Fall back to traditional method
            
            # Fallback to traditional sentiment analysis with bias detection
            # XLM-RoBERTa model is already loaded at startup
            # VADER sentiment analysis
            vader_scores = self.vader.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            # XLM-RoBERTa sentiment analysis
            xlm_sentiment = 'Neutral'
            xlm_score = 0.0
            xlm_confidence = 0.5
            
            if hasattr(self, 'xlm_roberta_sentiment') and self.xlm_roberta_sentiment:
                try:
                    xlm_result = self.xlm_roberta_sentiment(text)
                    if xlm_result:
                        # Handle different model output formats
                        if isinstance(xlm_result, list) and len(xlm_result) > 0:
                            result = xlm_result[0]
                            label = result.get('label', 'NEUTRAL').upper()
                            xlm_confidence = result.get('score', 0.5)
                            
                            # Map different label formats to sentiment
                            if 'POSITIVE' in label or 'POS' in label or label == 'LABEL_2':
                                xlm_sentiment = 'Positive'
                                xlm_score = xlm_confidence
                            elif 'NEGATIVE' in label or 'NEG' in label or label == 'LABEL_0':
                                xlm_sentiment = 'Negative'
                                xlm_score = -xlm_confidence
                            else:  # NEUTRAL or LABEL_1
                                xlm_sentiment = 'Neutral'
                                xlm_score = 0.0
                                
                except Exception as e:
                    print(f"[WARNING] XLM-RoBERTa sentiment analysis failed: {e}")
            
            # Combine VADER and XLM-RoBERTa scores (weighted average)
            vader_weight = 0.4
            xlm_weight = 0.6
            
            # Convert VADER compound to -1 to 1 scale if needed
            vader_normalized = max(-1, min(1, vader_compound))
            xlm_normalized = max(-1, min(1, xlm_score))
            
            combined_score = (vader_normalized * vader_weight) + (xlm_normalized * xlm_weight)
            
            # Classify sentiment based on combined score
            if combined_score >= 0.05:
                final_sentiment = 'Positive'
            elif combined_score <= -0.05:
                final_sentiment = 'Negative'
            else:
                final_sentiment = 'Neutral'
            
            # Convert to numerical score (-3 to +3 scale)
            numerical_score = combined_score * 3
            
            # Create initial result
            result = {
                'sentiment': final_sentiment,
                'score': combined_score,
                'numerical_score': numerical_score,
                'confidence': max(xlm_confidence, abs(vader_compound)),
                'vader_score': vader_compound,
                'xlm_roberta_sentiment': xlm_sentiment,
                'xlm_roberta_score': xlm_score,
                'method': 'combined_vader_xlm_roberta',
                'text': text
            }
            
            # Apply bias detection and correction for fallback method
            bias_corrected_result = bias_mitigator.postprocess_sentiment_with_bias_correction(text, result)
            
            # Add phase classification
            phase, _ = self.classify_time_period(text)
            bias_corrected_result['phase'] = phase
            
            # Log bias detection results if present
            if 'bias_analysis' in bias_corrected_result and bias_corrected_result['bias_analysis']['potential_bias_detected']:
                bias_types = ', '.join(bias_corrected_result['bias_analysis']['bias_types'])
                print(f"[BIAS DETECTED] {bias_types} | Severity: {bias_corrected_result['bias_analysis']['bias_severity']:.2f}")
            
            return bias_corrected_result
            
            # Use enhanced phase classification
            phase, _ = self.classify_time_period(text)
            
            return {
                'sentiment': final_sentiment,
                'score': combined_score,
                'numerical_score': numerical_score,
                'confidence': max(xlm_confidence, abs(vader_compound)),
                'phase': phase,
                'method': 'vader_xlm_roberta',
                'vader_score': vader_compound,
                'xlm_sentiment': xlm_sentiment,
                'xlm_score': xlm_score,
                'text': text
            }
            
        except Exception as e:
            print(f"❌ Sentiment analysis error: {e}")
            return {
                'sentiment': 'Neutral',
                'score': 0.0,
                'numerical_score': 0.0,
                'confidence': 0.5,
                'phase': 'during',
                'method': 'error_fallback',
                'text': text
            }
            phase, _ = self.classify_time_period(text, debug=True)
            
            return {
                'sentiment': final_sentiment,
                'score': combined_score,
                'numerical_score': numerical_score,
                'confidence': abs(combined_score),
                'phase': phase,
                'vader_detailed': vader_scores,
                'xlm_sentiment': xlm_sentiment,
                'xlm_score': xlm_score,
                'xlm_confidence': xlm_confidence,
                'model_used': 'XLM-RoBERTa + VADER'
            }
            
        except Exception as e:
            print(f"❌ Sentiment analysis error: {e}")
            return {
                'sentiment': 'Neutral',
                'score': 0.0,
                'numerical_score': 0.0,
                'confidence': 0.5,
                'phase': 'during',
                'error': str(e)
            }

    def run_analysis(self, audio_file_path):
        """
        Run comprehensive analysis on audio file
        """
        print(f"\n🎯 Starting comprehensive analysis of: {audio_file_path}")
        print("=" * 70)
        
        # Reset conversation history for new analysis
        if hasattr(self, 'contextual_analyzer') and self.contextual_analyzer:
            self.contextual_analyzer.reset_conversation_history()
            print("🔄 Reset conversation history for contextual analysis")
        
        try:
            # 1. Transcription
            print("📝 Transcribing audio...")
            transcription_result = self.transcribe_audio(audio_file_path)
            transcript = transcription_result.get('text', '')
            
            if not transcript:
                print("❌ Error: Failed to transcribe audio")
                return None
            
            print(f"✅ Transcription completed ({len(transcript)} characters)")
            
            # 2. Speaker Diarization (optional)
            print("\n🎭 Performing speaker diarization...")
            diarization_result = self.perform_speaker_diarization(audio_file_path)
            
            # 3. Segment the transcript
            print("\n🔄 Segmenting transcript...")
            segments = self.segment_transcript(transcript)
            print(f"✅ Created {len(segments)} segments")
            
            # 3. Classify speakers and sentiments
            print("\n🎭 Analyzing speakers and sentiments...")
            all_sentiments = []
            patient_sentiments = []
            
            for i, segment in enumerate(segments, 1):
                print(f"  Processing segment {i}/{len(segments)}...", end='\r')
                
                # Speaker classification
                speaker_result = self.classify_speaker_with_distilbert(segment)
                
                # Sentiment analysis with speaker context
                sentiment_result = self.analyze_sentiment(segment, speaker=speaker_result['speaker'])
                
                # Combine results
                segment_analysis = {
                    'segment_id': i,
                    'text': segment,
                    'speaker': speaker_result['speaker'],
                    'speaker_confidence': speaker_result['confidence'],
                    'sentiment': sentiment_result['sentiment'],
                    'score': sentiment_result.get('score', sentiment_result.get('numerical_score', 0)),
                    'numerical_score': sentiment_result.get('numerical_score', sentiment_result.get('score', 0)),
                    'confidence': sentiment_result['confidence'],
                    'phase': sentiment_result.get('phase', 'during'),
                    'method': sentiment_result.get('method', 'unknown')
                }
                
                all_sentiments.append(segment_analysis)
                
                # Collect patient sentiments for satisfaction analysis
                if speaker_result['speaker'] == 'PATIENT':
                    patient_sentiments.append(segment_analysis)
            
            print(f"\n✅ Analyzed {len(all_sentiments)} segments ({len(patient_sentiments)} patient segments)")
            
            # 4. Calculate patient satisfaction
            print("\n📊 Calculating patient satisfaction...")
            satisfaction_results = self.calculate_patient_satisfaction_score(patient_sentiments)
            
            # 5. Analyze sentiment trends
            print("\n📈 Analyzing sentiment trends...")
            trend_analysis = self.analyze_sentiment_trends(patient_sentiments)
            
            # 6. Generate conversation summary
            print("\n📋 Generating conversation summary...")
            conversation_summary = self.generate_conversation_summary(
                patient_sentiments, all_sentiments, satisfaction_results
            )
            
            # 7. Compile final results
            final_results = {
                'audio_file': audio_file_path,
                'transcription': {
                    'full_text': transcript,
                    'total_segments': len(segments),
                    'duration_estimate': f"{len(transcript) / 150:.1f} minutes"  # Approximate
                },
                'speaker_analysis': {
                    'total_segments': len(all_sentiments),
                    'patient_segments': len(patient_sentiments),
                    'doctor_segments': len(all_sentiments) - len(patient_sentiments)
                },
                'diarization_analysis': diarization_result if diarization_result else {
                    'available': False,
                    'der_estimate': 'N/A',
                    'unique_speakers': 'N/A',
                    'note': 'Diarization not performed'
                },
                'sentiment_analysis': {
                    'all_segments': all_sentiments,
                    'patient_segments': patient_sentiments,
                    'satisfaction_score': satisfaction_results['satisfaction_score'],
                    'satisfaction_level': satisfaction_results['satisfaction_level'],
                    'satisfaction_emoji': satisfaction_results['level_emoji']
                },
                'trend_analysis': trend_analysis,
                'conversation_summary': conversation_summary,
                'analysis_metadata': {
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'models_used': {
                        'transcription': 'Whisper',
                        'sentiment': 'XLM-RoBERTa + VADER',
                        'speaker_classification': 'DistilBERT Fine-tuned',
                        'phase_detection': 'Multi-layer Classification'
                    },
                    'hipaa_compliance': 'PHI Scrubbing Applied',
                    'multilingual_support': 'Enabled'
                }
            }
            
            print("\n🎉 Analysis completed successfully!")
            return final_results
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

def print_analysis_results(results):
    """
    Print simplified analysis results showing only essential information
    """
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("MEDICAL CONVERSATION ANALYSIS")
    print("="*80)
    
    # Audio file info
    print(f"Audio File: {results['audio_file']}")
    
    # Patient Satisfaction Score (from patient sentiments only)
    sentiment_data = results['sentiment_analysis']
    print(f"\nPATIENT SATISFACTION SCORE: {sentiment_data['satisfaction_score']}/100")
    print(f"   Level: {sentiment_data['satisfaction_level']}")
    
    # Diarization Error Rate (DER)
    if 'diarization_analysis' in results and results['diarization_analysis']:
        der = results['diarization_analysis'].get('der_estimate', 'N/A')
        print(f"\nDIARIZATION ERROR RATE (DER): {der}")
        if isinstance(der, float):
            if der < 0.1:
                print(f"   Quality: Excellent (DER < 10%)")
            elif der < 0.2:
                print(f"   Quality: Good (DER < 20%)")
            else:
                print(f"   Quality: Fair (DER ≥ 20%)")
    
    # Transcribed text with speaker classification and patient sentiment analysis
    print(f"\n� TRANSCRIBED CONVERSATION:")
    print("-" * 80)
    
    all_segments = sentiment_data['all_segments']
    patient_segments = sentiment_data['patient_segments']
    
    # Create a lookup for patient sentiment data
    patient_sentiment_lookup = {}
    for p_seg in patient_segments:
        patient_sentiment_lookup[p_seg['segment_id']] = p_seg
    
    for segment in all_segments:
        speaker = "DOCTOR" if segment['speaker'] == 'DOCTOR' else "PATIENT"
        text = segment['text']
        
        # Basic display for all segments
        print(f"\n{speaker}: {text}")
        
        # Additional sentiment info only for patient segments
        if segment['speaker'] == 'PATIENT' and segment['segment_id'] in patient_sentiment_lookup:
            p_data = patient_sentiment_lookup[segment['segment_id']]
            sentiment = p_data.get('sentiment', 'Unknown')
            score = p_data.get('numerical_score', 0)
            phase = p_data.get('phase', 'during')
            
            print(f"   └─ Sentiment: {sentiment} | Score: {score:.2f} | Phase: {phase}")
    
    print("\n" + "="*80)
    print("End of Analysis")
    print("="*80)

def select_audio_file():
    """
    Interactive file selection for audio analysis
    """
    print("🎵 Audio File Selection")
    print("=" * 40)
    
    # First, show available audio files in current directory
    current_dir = os.getcwd()
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
    
    audio_files = []
    for file in os.listdir(current_dir):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(file)
    
    if audio_files:
        print(f"\n📂 Available audio files in {current_dir}:")
        for i, file in enumerate(audio_files, 1):
            try:
                file_size = os.path.getsize(file) / (1024*1024)  # MB
                print(f"  {i}. {file} ({file_size:.1f} MB)")
            except:
                print(f"  {i}. {file}")
    
    # Prompt for file path
    while True:
        if audio_files:
            print(f"\n💡 Tip: You can enter a number (1-{len(audio_files)}) to select from the list above,")
            print("    or enter the full path to any audio file, or type 'exit' to quit.")

        user_input = ("C:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\Recording (3).wav").strip()

        if not user_input:
            print("❌ Please enter a file path, number, or 'exit'")
            continue
        
        # Check if user wants to exit
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            return None
        
        # Check if user entered a number to select from list
        if user_input.isdigit() and audio_files:
            file_index = int(user_input) - 1
            if 0 <= file_index < len(audio_files):
                selected_file = os.path.join(current_dir, audio_files[file_index])
                print(f"✅ Selected: {selected_file}")
                return selected_file
            else:
                print(f"❌ Invalid number. Please enter 1-{len(audio_files)}")
                continue
        
        # Check if user entered a file path
        file_path = user_input.strip('"\'')  # Remove quotes if present
        
        if os.path.exists(file_path):
            if any(file_path.lower().endswith(ext) for ext in audio_extensions):
                print(f"✅ Selected: {file_path}")
                return os.path.abspath(file_path)
            else:
                print("❌ File doesn't appear to be an audio file.")
                print(f"   Supported formats: {', '.join(audio_extensions)}")
                continue
        else:
            print(f"❌ File not found: {file_path}")
            continue

def main():
    """
    Main function for comprehensive audio sentiment analysis using cached XLM-RoBERTa
    """
    # Set offline mode for cached model loading
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    print("🎯 ENHANCED MEDICAL CONVERSATION ANALYZER")
    print("=" * 60)
    print("🚀 Advanced AI-powered analysis with XLM-RoBERTa sentiment model")
    print("✨ Features: Cached Model Loading • Sentiment Analysis • Phase Classification")
    print("📊 Patient Satisfaction Scoring • Speaker Diarization • HIPAA Compliance")
    print("=" * 60)
    
    # Target audio file - CAR0001 as requested
    audio_file_path = "C:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\car0001_avFikfGD.wav"
    
    # Fallback audio files to try if the main file is not found
    fallback_files = [
        "CAR0001.wav",
        "car0001_avFikfGD.wav",
        "Recording.wav", 
        "Recording (2).wav",
        "Recording (3).wav"
    ]
    
    # Check if the target audio file exists
    if not os.path.exists(audio_file_path):
        print(f"🔍 Primary audio file not found: {audio_file_path}")
        
        # Try fallback files
        found_file = None
        for fallback in fallback_files:
            full_path = os.path.join(os.getcwd(), fallback)
            if os.path.exists(full_path):
                audio_file_path = full_path
                found_file = fallback
                break
        
        if found_file:
            print(f"📁 Using fallback audio file: {found_file}")
        else:
            # Try to find any audio file in the current directory
            current_dir = os.getcwd()
            audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
            
            audio_files = []
            for file in os.listdir(current_dir):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(file)
            
            if audio_files:
                audio_file_path = os.path.join(current_dir, audio_files[0])
                print(f"📁 No target file found, using first available: {audio_files[0]}")
            else:
                print("❌ No audio files found in the current directory!")
                print("   Please ensure the audio file exists or update the file path.")
                print(f"   Expected file: car0001_avFikfGD.wav")
                return
    
    print(f"🎵 Processing audio file: {os.path.basename(audio_file_path)}")
    print(f"📂 Full path: {audio_file_path}")
    
    # Check file size and properties
    try:
        file_size = os.path.getsize(audio_file_path)
        print(f"📊 File size: {file_size / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"⚠️  Warning: Could not get file size: {e}")
    
    # Initialize analyzer with cached models
    print("\n🔄 Initializing Enhanced Batch Audio Analyzer...")
    print("⚡ Loading all models from cache for immediate availability...")
    
    try:
        analyzer = BatchAudioAnalyzer()
        print("✅ Analyzer initialized successfully with cached models!")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        print("📋 Full error traceback:")
        import traceback
        traceback.print_exc()
        return
    
    # Run comprehensive analysis
    print(f"\n🎯 Starting comprehensive analysis...")
    print("🔍 Processing stages: Transcription → Sentiment Analysis → Phase Classification → Satisfaction Scoring")
    
    try:
        results = analyzer.run_analysis(audio_file_path)
        
        if results and 'error' not in results:
            print("\n✅ Analysis completed successfully!")
            
            # Display comprehensive results
            print_analysis_results(results)
            
            # Save results with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"car0001_analysis_results_{timestamp}.json"
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                    
                print(f"💾 Results saved to: {output_file}")
                
                # Generate bias monitoring report
                print(f"\n📊 Generating bias monitoring report...")
                bias_report = bias_detector.generate_bias_report()
                
                bias_report_file = f"bias_analysis_report_{timestamp}.json"
                with open(bias_report_file, 'w', encoding='utf-8') as f:
                    json.dump(bias_report, f, indent=2, ensure_ascii=False, default=str)
                    
                print(f"📈 Bias analysis report saved to: {bias_report_file}")
                
                # Display bias summary
                print_bias_summary(bias_report)
                
            except Exception as e:
                print(f"⚠️ Warning: Could not save results: {e}")
                
        else:
            print("❌ Analysis failed or returned no results")
            if results and 'error' in results:
                print(f"Error: {results['error']}")
                
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        print("📋 Full error traceback:")
        import traceback
        traceback.print_exc()

def print_bias_summary(bias_report):
    """
    Display bias analysis summary
    """
    print("\n" + "="*70)
    print("🛡️ BIAS ANALYSIS SUMMARY")
    print("="*70)
    
    summary = bias_report.get('summary', {})
    metrics = bias_report.get('detailed_metrics', {})
    
    print(f"📊 Overall Bias Detection Rate: {summary.get('overall_bias_rate', 0):.1%}")
    print(f"🎯 Most Common Bias Type: {summary.get('most_common_bias_type', 'None')}")
    print(f"📈 Total Incidents Logged: {summary.get('total_incidents', 0)}")
    
    if 'bias_detection_rate' in metrics:
        rates = metrics['bias_detection_rate']
        print(f"\n📋 Detailed Bias Rates:")
        print(f"   👥 Gender Bias: {rates.get('gender_bias_rate', 0):.1%}")
        print(f"   🌍 Ethnicity/Cultural Bias: {rates.get('ethnicity_bias_rate', 0):.1%}")
        print(f"   💰 Social Class Bias: {rates.get('social_class_bias_rate', 0):.1%}")
    
    # Show high-severity incidents
    high_severity = bias_report.get('recent_high_severity_incidents', [])
    if high_severity:
        print(f"\n⚠️ High-Severity Bias Incidents ({len(high_severity)}):")
        for i, incident in enumerate(high_severity[:3], 1):  # Show top 3
            bias_types = ', '.join(incident['bias_analysis']['bias_types'])
            severity = incident['bias_analysis']['bias_severity']
            print(f"   {i}. {bias_types} (Severity: {severity:.2f})")
            print(f"      Text: \"{incident['text_sample']}\"")
    
    # Show recommendations
    recommendations = bias_report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Bias Mitigation Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. [{rec['priority'].upper()}] {rec['action']}")
    else:
        print(f"\n✅ No major bias issues detected - system performing well!")
    
    print("="*70)

if __name__ == "__main__":
    main()
