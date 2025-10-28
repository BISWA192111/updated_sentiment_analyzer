"""
Real-Time Patient Experience (PX) Score Platform
================================================

A comprehensive platform for real-time patient experience monitoring with:
- Live PX score calculation (0-100 scale)
- Multi-patient dashboard
- Alert system for low scores
- Historical tracking and analytics
- Integration with existing sentiment analysis models

Author: PX Score Team
Date: October 2025
"""

import os
import sys
import json
import uuid
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

# Import existing sentiment analysis capabilities
try:
    from new_sentiment import BatchAudioAnalyzer
    from patient_satisfaction_predictor import PatientSatisfactionPredictor
    MODELS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Sentiment models not available. Using mock implementation.")
    MODELS_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels for PX scores"""
    CRITICAL = "CRITICAL"  # PX Score < 40
    WARNING = "WARNING"    # PX Score 40-60
    NORMAL = "NORMAL"      # PX Score 60-80
    EXCELLENT = "EXCELLENT"  # PX Score > 80


@dataclass
class PatientInteraction:
    """Single patient interaction data point"""
    interaction_id: str
    patient_id: str
    timestamp: datetime
    text: str
    sentiment: str
    sentiment_score: float
    px_score: float
    speaker: str  # 'PATIENT' or 'DOCTOR'
    phase: str  # 'BEFORE', 'DURING', 'AFTER'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PatientSession:
    """Complete patient session with aggregated PX score"""
    session_id: str
    patient_id: str
    patient_name: str
    start_time: datetime
    last_update: datetime
    current_px_score: float
    interaction_count: int
    sentiment_distribution: Dict[str, int]
    alert_level: AlertLevel
    notes: str = ""
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['last_update'] = self.last_update.isoformat()
        data['alert_level'] = self.alert_level.value
        return data


class PXScoreCalculator:
    """
    Calculates Patient Experience (PX) scores from sentiment analysis
    
    PX Score Formula:
    - Base score from sentiment (0-100)
    - Weighted by recency (recent interactions matter more)
    - Adjusted by conversation phase
    - Calibrated using historical data
    """
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.satisfaction_predictor = None
        self._initialize_models()
        
        # PX Score weights
        self.weights = {
            'sentiment_positive': 1.0,
            'sentiment_neutral': 0.5,
            'sentiment_negative': 0.0,
            'recency_decay': 0.95,  # Exponential decay for older interactions
            'phase_multiplier': {
                'BEFORE': 1.1,  # Pre-treatment concerns matter more
                'DURING': 1.0,
                'AFTER': 1.05   # Post-treatment satisfaction matters more
            }
        }
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        if MODELS_AVAILABLE:
            try:
                print("🔧 Initializing PX Score calculation models...")
                self.sentiment_analyzer = BatchAudioAnalyzer()
                self.satisfaction_predictor = PatientSatisfactionPredictor()
                print("✅ Models loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading models: {e}")
                print("Using mock scoring instead")
        else:
            print("📝 Using mock PX score calculation")
    
    def calculate_interaction_score(self, text: str, phase: str = 'DURING') -> Tuple[str, float, float]:
        """
        Calculate PX score for a single interaction
        
        Args:
            text: Patient statement or conversation text
            phase: Treatment phase (BEFORE/DURING/AFTER)
        
        Returns:
            Tuple of (sentiment_label, sentiment_confidence, px_score)
        """
        if self.satisfaction_predictor:
            try:
                # Use existing satisfaction predictor
                sentiment_data = self.satisfaction_predictor.analyze_statement_sentiment(text)
                sentiment_label = sentiment_data.get('label', 'Neutral')
                sentiment_confidence = sentiment_data.get('confidence', 0.5)
                
                # Convert sentiment to PX score (0-100)
                base_score = self._sentiment_to_px_score(sentiment_label, sentiment_confidence)
                
                # Apply phase multiplier
                phase_multiplier = self.weights['phase_multiplier'].get(phase, 1.0)
                px_score = min(100, base_score * phase_multiplier)
                
                return sentiment_label, sentiment_confidence, px_score
                
            except Exception as e:
                print(f"⚠️ Error in score calculation: {e}")
                return self._mock_score(text)
        else:
            return self._mock_score(text)
    
    def _sentiment_to_px_score(self, sentiment: str, confidence: float) -> float:
        """Convert sentiment to PX score (0-100)"""
        sentiment = sentiment.upper()
        
        if 'POSITIVE' in sentiment or sentiment == 'POSITIVE':
            # Positive sentiment: 60-100 based on confidence
            base = 60 + (40 * confidence)
        elif 'NEGATIVE' in sentiment or sentiment == 'NEGATIVE':
            # Negative sentiment: 0-40 based on confidence
            base = 40 * (1 - confidence)
        else:
            # Neutral sentiment: 40-60
            base = 50
        
        return round(base, 2)
    
    def calculate_session_score(self, interactions: List[PatientInteraction]) -> float:
        """
        Calculate aggregated PX score for entire session
        
        Uses weighted average with recency decay:
        - Recent interactions weighted more heavily
        - Exponential decay for older interactions
        """
        if not interactions:
            return 50.0  # Default neutral score
        
        # Sort by timestamp (newest first)
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp, reverse=True)
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for i, interaction in enumerate(sorted_interactions):
            # Calculate recency weight (exponential decay)
            weight = self.weights['recency_decay'] ** i
            
            weighted_sum += interaction.px_score * weight
            weight_total += weight
        
        session_score = weighted_sum / weight_total if weight_total > 0 else 50.0
        return round(session_score, 2)
    
    def _mock_score(self, text: str) -> Tuple[str, float, float]:
        """Mock scoring for testing when models unavailable"""
        # Simple keyword-based mock
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'better', 'thank', 'happy', 'excellent']
        negative_words = ['pain', 'hurt', 'bad', 'worse', 'terrible', 'awful']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'Positive', 0.75, 75.0
        elif neg_count > pos_count:
            return 'Negative', 0.75, 25.0
        else:
            return 'Neutral', 0.5, 50.0


class PXScoreDatabase:
    """
    SQLite database for storing patient interactions and sessions
    """
    
    def __init__(self, db_path: str = "px_score_data.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Patient sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_sessions (
                session_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                start_time TEXT NOT NULL,
                last_update TEXT NOT NULL,
                current_px_score REAL NOT NULL,
                interaction_count INTEGER DEFAULT 0,
                alert_level TEXT DEFAULT 'NORMAL',
                notes TEXT,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        # Patient interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_interactions (
                interaction_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                px_score REAL NOT NULL,
                speaker TEXT NOT NULL,
                phase TEXT DEFAULT 'DURING',
                FOREIGN KEY (session_id) REFERENCES patient_sessions(session_id)
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS px_alerts (
                alert_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                alert_level TEXT NOT NULL,
                px_score REAL NOT NULL,
                message TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES patient_sessions(session_id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_active ON patient_sessions(is_active)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_session ON patient_interactions(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON px_alerts(acknowledged)')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    def create_session(self, patient_id: str, patient_name: str) -> str:
        """Create new patient session"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patient_sessions 
            (session_id, patient_id, patient_name, start_time, last_update, current_px_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, patient_id, patient_name, now, now, 50.0))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def add_interaction(self, interaction: PatientInteraction):
        """Add new patient interaction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patient_interactions 
            (interaction_id, session_id, patient_id, timestamp, text, sentiment, 
             sentiment_score, px_score, speaker, phase)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            interaction.interaction_id,
            interaction.patient_id,  # Using patient_id as session_id for now
            interaction.patient_id,
            interaction.timestamp.isoformat(),
            interaction.text,
            interaction.sentiment,
            interaction.sentiment_score,
            interaction.px_score,
            interaction.speaker,
            interaction.phase
        ))
        
        conn.commit()
        conn.close()
    
    def update_session_score(self, session_id: str, px_score: float, alert_level: AlertLevel):
        """Update session PX score"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE patient_sessions
            SET current_px_score = ?,
                last_update = ?,
                alert_level = ?,
                interaction_count = interaction_count + 1
            WHERE session_id = ?
        ''', (px_score, datetime.now().isoformat(), alert_level.value, session_id))
        
        conn.commit()
        conn.close()
    
    def get_active_sessions(self) -> List[Dict]:
        """Get all active patient sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, patient_id, patient_name, start_time, last_update,
                   current_px_score, interaction_count, alert_level, notes
            FROM patient_sessions
            WHERE is_active = 1
            ORDER BY last_update DESC
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        sessions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return sessions
    
    def get_session_interactions(self, session_id: str) -> List[Dict]:
        """Get all interactions for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT interaction_id, timestamp, text, sentiment, sentiment_score,
                   px_score, speaker, phase
            FROM patient_interactions
            WHERE session_id = ?
            ORDER BY timestamp ASC
        ''', (session_id,))
        
        columns = [desc[0] for desc in cursor.description]
        interactions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return interactions
    
    def create_alert(self, session_id: str, patient_id: str, alert_level: AlertLevel, 
                     px_score: float, message: str):
        """Create new alert for low PX score"""
        alert_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO px_alerts
            (alert_id, session_id, patient_id, timestamp, alert_level, px_score, message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (alert_id, session_id, patient_id, datetime.now().isoformat(), 
              alert_level.value, px_score, message))
        
        conn.commit()
        conn.close()
        
        return alert_id
    
    def get_unacknowledged_alerts(self) -> List[Dict]:
        """Get all unacknowledged alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.alert_id, a.session_id, a.patient_id, a.timestamp,
                   a.alert_level, a.px_score, a.message, s.patient_name
            FROM px_alerts a
            JOIN patient_sessions s ON a.session_id = s.session_id
            WHERE a.acknowledged = 0
            ORDER BY a.timestamp DESC
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return alerts


class AlertManager:
    """
    Manages alerts for low PX scores
    """
    
    def __init__(self, database: PXScoreDatabase):
        self.db = database
        self.alert_thresholds = {
            AlertLevel.CRITICAL: 40,
            AlertLevel.WARNING: 60,
            AlertLevel.NORMAL: 80,
            AlertLevel.EXCELLENT: 100
        }
        self.alert_callbacks = []
    
    def determine_alert_level(self, px_score: float) -> AlertLevel:
        """Determine alert level based on PX score"""
        if px_score < self.alert_thresholds[AlertLevel.CRITICAL]:
            return AlertLevel.CRITICAL
        elif px_score < self.alert_thresholds[AlertLevel.WARNING]:
            return AlertLevel.WARNING
        elif px_score < self.alert_thresholds[AlertLevel.NORMAL]:
            return AlertLevel.NORMAL
        else:
            return AlertLevel.EXCELLENT
    
    def check_and_create_alert(self, session_id: str, patient_id: str, 
                                patient_name: str, px_score: float) -> Optional[str]:
        """Check if alert should be created and create if necessary"""
        alert_level = self.determine_alert_level(px_score)
        
        # Only create alerts for WARNING and CRITICAL
        if alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
            message = self._generate_alert_message(patient_name, px_score, alert_level)
            alert_id = self.db.create_alert(session_id, patient_id, alert_level, px_score, message)
            
            # Trigger alert callbacks
            self._trigger_callbacks(alert_level, patient_name, px_score, message)
            
            return alert_id
        
        return None
    
    def _generate_alert_message(self, patient_name: str, px_score: float, 
                                 alert_level: AlertLevel) -> str:
        """Generate alert message"""
        if alert_level == AlertLevel.CRITICAL:
            return f"🚨 CRITICAL: Patient {patient_name} has very low PX score ({px_score:.1f}). Immediate attention required!"
        elif alert_level == AlertLevel.WARNING:
            return f"⚠️ WARNING: Patient {patient_name} has low PX score ({px_score:.1f}). Check-in recommended."
        else:
            return f"ℹ️ Patient {patient_name} PX score: {px_score:.1f}"
    
    def register_callback(self, callback):
        """Register callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def _trigger_callbacks(self, alert_level: AlertLevel, patient_name: str, 
                           px_score: float, message: str):
        """Trigger all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_level, patient_name, px_score, message)
            except Exception as e:
                print(f"Error in alert callback: {e}")


class PXScorePlatform:
    """
    Main PX Score Platform
    Coordinates all components for real-time patient experience monitoring
    """
    
    def __init__(self, db_path: str = "px_score_data.db"):
        print("🏥 Initializing PX Score Platform...")
        
        self.calculator = PXScoreCalculator()
        self.database = PXScoreDatabase(db_path)
        self.alert_manager = AlertManager(self.database)
        
        # Active sessions cache
        self.active_sessions: Dict[str, List[PatientInteraction]] = {}
        
        print("✅ PX Score Platform ready!")
    
    def start_patient_session(self, patient_id: str, patient_name: str) -> str:
        """Start new patient session"""
        session_id = self.database.create_session(patient_id, patient_name)
        self.active_sessions[session_id] = []
        
        print(f"📋 Started session for {patient_name} (ID: {patient_id})")
        return session_id
    
    def process_patient_statement(self, session_id: str, patient_id: str, 
                                   text: str, speaker: str = 'PATIENT', 
                                   phase: str = 'DURING') -> Dict:
        """
        Process a patient statement and update PX score
        
        Returns:
            Dictionary with interaction details and updated PX score
        """
        # Calculate PX score for this interaction
        sentiment, confidence, px_score = self.calculator.calculate_interaction_score(text, phase)
        
        # Create interaction record
        interaction = PatientInteraction(
            interaction_id=str(uuid.uuid4()),
            patient_id=patient_id,
            timestamp=datetime.now(),
            text=text,
            sentiment=sentiment,
            sentiment_score=confidence,
            px_score=px_score,
            speaker=speaker,
            phase=phase
        )
        
        # Store interaction
        self.database.add_interaction(interaction)
        
        # Update session cache
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        self.active_sessions[session_id].append(interaction)
        
        # Calculate updated session score
        session_score = self.calculator.calculate_session_score(
            self.active_sessions[session_id]
        )
        
        # Determine alert level
        alert_level = self.alert_manager.determine_alert_level(session_score)
        
        # Update session in database
        self.database.update_session_score(session_id, session_score, alert_level)
        
        # Check for alerts
        sessions = self.database.get_active_sessions()
        patient_name = next((s['patient_name'] for s in sessions if s['session_id'] == session_id), 'Unknown')
        alert_id = self.alert_manager.check_and_create_alert(
            session_id, patient_id, patient_name, session_score
        )
        
        return {
            'interaction_id': interaction.interaction_id,
            'sentiment': sentiment,
            'sentiment_confidence': confidence,
            'interaction_px_score': px_score,
            'session_px_score': session_score,
            'alert_level': alert_level.value,
            'alert_created': alert_id is not None,
            'alert_id': alert_id
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get all data needed for dashboard display"""
        active_sessions = self.database.get_active_sessions()
        alerts = self.database.get_unacknowledged_alerts()
        
        # Calculate summary statistics
        if active_sessions:
            avg_score = np.mean([s['current_px_score'] for s in active_sessions])
            critical_count = sum(1 for s in active_sessions if s['alert_level'] == 'CRITICAL')
            warning_count = sum(1 for s in active_sessions if s['alert_level'] == 'WARNING')
        else:
            avg_score = 0
            critical_count = 0
            warning_count = 0
        
        return {
            'active_sessions': active_sessions,
            'unacknowledged_alerts': alerts,
            'summary': {
                'total_active_patients': len(active_sessions),
                'average_px_score': round(avg_score, 2),
                'critical_alerts': critical_count,
                'warning_alerts': warning_count
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_patient_history(self, session_id: str) -> Dict:
        """Get detailed history for a specific patient session"""
        interactions = self.database.get_session_interactions(session_id)
        sessions = self.database.get_active_sessions()
        session_info = next((s for s in sessions if s['session_id'] == session_id), None)
        
        if not session_info:
            return {'error': 'Session not found'}
        
        return {
            'session_info': session_info,
            'interactions': interactions,
            'interaction_count': len(interactions)
        }


def demo_px_platform():
    """Demonstration of PX Score Platform"""
    print("\n" + "="*60)
    print(" PX SCORE PLATFORM DEMO")
    print("="*60 + "\n")
    
    # Initialize platform
    platform = PXScorePlatform()
    
    # Register alert callback
    def alert_callback(level, patient_name, score, message):
        print(f"\n ALERT: {message}\n")
    
    platform.alert_manager.register_callback(alert_callback)
    
    # Simulate patient sessions
    print(" Creating patient sessions...\n")
    
    # Patient 1: Positive experience
    session1 = platform.start_patient_session("P001", "John Smith")
    
    statements1 = [
        ("The doctor was very helpful and explained everything clearly.", "PATIENT", "DURING"),
        ("I'm feeling much better after the treatment.", "PATIENT", "AFTER"),
        ("Thank you for the excellent care!", "PATIENT", "AFTER")
    ]
    
    print(f"\n--- Patient: John Smith ---")
    for text, speaker, phase in statements1:
        result = platform.process_patient_statement(session1, "P001", text, speaker, phase)
        print(f" {text[:50]}...")
        print(f"   Sentiment: {result['sentiment']} | PX Score: {result['session_px_score']:.1f} | Alert: {result['alert_level']}")
    
    # Patient 2: Negative experience (will trigger alerts)
    session2 = platform.start_patient_session("P002", "Jane Doe")
    
    statements2 = [
        ("I'm in a lot of pain and feel like nobody is listening.", "PATIENT", "DURING"),
        ("The wait time was terrible and I'm still hurting.", "PATIENT", "DURING"),
        ("This treatment isn't helping at all.", "PATIENT", "AFTER")
    ]
    
    print(f"\n--- Patient: Jane Doe ---")
    for text, speaker, phase in statements2:
        result = platform.process_patient_statement(session2, "P002", text, speaker, phase)
        print(f" {text[:50]}...")
        print(f"   Sentiment: {result['sentiment']} | PX Score: {result['session_px_score']:.1f} | Alert: {result['alert_level']}")
    
    # Patient 3: Mixed experience
    session3 = platform.start_patient_session("P003", "Bob Johnson")
    
    statements3 = [
        ("I'm nervous about this procedure.", "PATIENT", "BEFORE"),
        ("The staff is friendly but I'm still worried.", "PATIENT", "BEFORE"),
        ("That went better than expected.", "PATIENT", "AFTER")
    ]
    
    print(f"\n--- Patient: Bob Johnson ---")
    for text, speaker, phase in statements3:
        result = platform.process_patient_statement(session3, "P003", text, speaker, phase)
        print(f" {text[:50]}...")
        print(f"   Sentiment: {result['sentiment']} | PX Score: {result['session_px_score']:.1f} | Alert: {result['alert_level']}")
    
    # Display dashboard
    print("\n" + "="*60)
    print(" DASHBOARD SUMMARY")
    print("="*60 + "\n")
    
    dashboard = platform.get_dashboard_data()
    summary = dashboard['summary']
    
    print(f"Active Patients: {summary['total_active_patients']}")
    print(f"Average PX Score: {summary['average_px_score']:.1f}")
    print(f"Critical Alerts: {summary['critical_alerts']}")
    print(f"Warning Alerts: {summary['warning_alerts']}")
    
    print("\n--- Active Patient Sessions ---")
    for session in dashboard['active_sessions']:
        alert_emoji = "" if session['alert_level'] == 'CRITICAL' else "⚠️" if session['alert_level'] == 'WARNING' else "✅"
        print(f"{alert_emoji} {session['patient_name']}: PX Score = {session['current_px_score']:.1f} ({session['interaction_count']} interactions)")
    
    print("\n--- Unacknowledged Alerts ---")
    for alert in dashboard['unacknowledged_alerts']:
        print(f"  {alert['message']}")
    
    print("\n" + "="*60)
    print(" Demo completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    demo_px_platform()
