#!/usr/bin/env python3
"""
SQLite Database for RLHF System

This module provides SQLite-based data persistence for the RLHF system,
replacing JSONL files with a proper relational database for better performance
and complex querying capabilities.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLHFSQLiteDB:
    """SQLite-based RLHF Database System"""
    
    def __init__(self, db_path: str = None):
        """Initialize SQLite database"""
        if db_path is None:
            # Default to project root/data/rlhf.db
            project_root = Path(__file__).resolve().parents[1]
            self.db_path = project_root / "data" / "rlhf.db"
        else:
            self.db_path = Path(db_path)
        
        # Ensure data directory exists
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Initialize database
        self._init_database()
        logger.info(f"Initialized SQLite database at {self.db_path}")
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Prompts table - stores generated prompts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    enhanced_prompt TEXT NOT NULL,
                    domain TEXT,
                    difficulty TEXT,
                    variation_type TEXT,
                    complexity_score REAL,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Completions table - stores model responses  
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS completions (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL,
                    completion_text TEXT NOT NULL,
                    model_provider TEXT,
                    model_name TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    generation_cost REAL,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prompt_id) REFERENCES prompts (id)
                )
            """)
            
            # Votes table - stores human preferences
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS votes (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL,
                    completion_a_id TEXT NOT NULL,
                    completion_b_id TEXT NOT NULL,
                    chosen_completion_id TEXT NOT NULL,
                    human_choice TEXT,  -- 'A' or 'B'
                    confidence REAL,
                    annotation TEXT,
                    is_model_vote BOOLEAN DEFAULT FALSE,
                    quality_metrics JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prompt_id) REFERENCES prompts (id),
                    FOREIGN KEY (completion_a_id) REFERENCES completions (id),
                    FOREIGN KEY (completion_b_id) REFERENCES completions (id),
                    FOREIGN KEY (chosen_completion_id) REFERENCES completions (id)
                )
            """)
            
            # Predictions table - stores model predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    vote_id TEXT,
                    prompt_id TEXT NOT NULL,
                    completion_a_id TEXT NOT NULL,
                    completion_b_id TEXT NOT NULL,
                    predicted_choice TEXT,  -- 'A' or 'B'
                    prediction_confidence REAL,
                    model_correct BOOLEAN,
                    raw_logits JSON,
                    raw_probabilities JSON,
                    predictor_model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (vote_id) REFERENCES votes (id),
                    FOREIGN KEY (prompt_id) REFERENCES prompts (id),
                    FOREIGN KEY (completion_a_id) REFERENCES completions (id),
                    FOREIGN KEY (completion_b_id) REFERENCES completions (id)
                )
            """)
            
            # Reflections table - stores meta-evaluation data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    id TEXT PRIMARY KEY,
                    vote_id TEXT,
                    prediction_id TEXT,
                    reflection_text TEXT,
                    analysis_type TEXT,
                    error_type TEXT,
                    confidence_gap REAL,
                    drift_indicators JSON,
                    improvement_suggestions TEXT,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (vote_id) REFERENCES votes (id),
                    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
                )
            """)
            
            # Model checkpoints table - tracks model evolution
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    model_path TEXT,
                    performance_metrics JSON,
                    training_data_size INTEGER,
                    training_duration_seconds REAL,
                    validation_accuracy REAL,
                    calibration_error REAL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_votes_prompt_id ON votes (prompt_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_votes_created_at ON votes (created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_completions_prompt_id ON completions (prompt_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_vote_id ON predictions (vote_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_vote_id ON reflections (vote_id)")
            
            conn.commit()
    
    def save_annotation(self, annotation_data: Dict[str, Any]) -> bool:
        """Save annotation through SQLite with proper relationships"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Generate IDs
                prompt_id = annotation_data.get("prompt_id", f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
                completion_a_id = f"comp_a_{prompt_id}"
                completion_b_id = f"comp_b_{prompt_id}"
                vote_id = f"vote_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                
                # 1. Save prompt
                cursor.execute("""
                    INSERT OR REPLACE INTO prompts 
                    (id, enhanced_prompt, domain, difficulty, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    prompt_id,
                    annotation_data.get("prompt", ""),
                    annotation_data.get("domain", "general"),
                    annotation_data.get("difficulty", "intermediate"),
                    json.dumps(annotation_data.get("prompt_metadata", {}))
                ))
                
                # 2. Save completions
                cursor.execute("""
                    INSERT OR REPLACE INTO completions 
                    (id, prompt_id, completion_text, model_provider, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    completion_a_id,
                    prompt_id,
                    annotation_data.get("completion_a", ""),
                    annotation_data.get("model_provider", "unknown"),
                    json.dumps({"position": "A"})
                ))
                
                cursor.execute("""
                    INSERT OR REPLACE INTO completions 
                    (id, prompt_id, completion_text, model_provider, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    completion_b_id,
                    prompt_id,
                    annotation_data.get("completion_b", ""),
                    annotation_data.get("model_provider", "unknown"),
                    json.dumps({"position": "B"})
                ))
                
                # 3. Save vote
                chosen_completion_id = completion_a_id if annotation_data.get("preference") == "Completion A" else completion_b_id
                human_choice = "A" if annotation_data.get("preference") == "Completion A" else "B"
                
                cursor.execute("""
                    INSERT OR REPLACE INTO votes 
                    (id, prompt_id, completion_a_id, completion_b_id, chosen_completion_id, 
                     human_choice, confidence, annotation, quality_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    vote_id,
                    prompt_id,
                    completion_a_id,
                    completion_b_id,
                    chosen_completion_id,
                    human_choice,
                    annotation_data.get("confidence", 0.8),
                    annotation_data.get("feedback", ""),
                    json.dumps(annotation_data.get("quality_metrics", {}))
                ))
                
                conn.commit()
                logger.info(f"Saved annotation {vote_id} to SQLite database")
                return True
                
        except Exception as e:
            logger.error(f"Error saving annotation to SQLite: {e}")
            return False
    
    def get_annotations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get annotations with all related data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT 
                        v.id as vote_id,
                        v.prompt_id,
                        p.enhanced_prompt as prompt,
                        ca.completion_text as completion_a,
                        cb.completion_text as completion_b,
                        v.human_choice,
                        v.confidence,
                        v.annotation,
                        v.created_at,
                        v.quality_metrics
                    FROM votes v
                    JOIN prompts p ON v.prompt_id = p.id
                    JOIN completions ca ON v.completion_a_id = ca.id
                    JOIN completions cb ON v.completion_b_id = cb.id
                    ORDER BY v.created_at DESC
                    LIMIT ?
                """
                
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
                
                columns = [desc[0] for desc in cursor.description]
                annotations = []
                
                for row in rows:
                    annotation = dict(zip(columns, row))
                    # Parse JSON fields
                    if annotation['quality_metrics']:
                        annotation['quality_metrics'] = json.loads(annotation['quality_metrics'])
                    annotations.append(annotation)
                
                return annotations
                
        except Exception as e:
            logger.error(f"Error getting annotations from SQLite: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                for table in ['prompts', 'completions', 'votes', 'predictions', 'reflections']:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"total_{table}"] = cursor.fetchone()[0]
                
                # Recent activity (last 24 hours)
                cursor.execute("""
                    SELECT COUNT(*) FROM votes 
                    WHERE created_at > datetime('now', '-1 day')
                """)
                stats["votes_last_24h"] = cursor.fetchone()[0]
                
                # Model accuracy (if predictions exist)
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN model_correct = 1 THEN 1 ELSE 0 END) as correct
                    FROM predictions
                """)
                result = cursor.fetchone()
                if result[0] > 0:
                    stats["model_accuracy"] = result[1] / result[0]
                else:
                    stats["model_accuracy"] = None
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting statistics from SQLite: {e}")
            return {}
    
    def get_table_info(self) -> Dict[str, List[str]]:
        """Get table schema information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                table_info = {}
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for table_name in [t[0] for t in tables]:
                    # Get column info for each table
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    table_info[table_name] = [col[1] for col in columns]  # col[1] is column name
                
                return table_info
                
        except Exception as e:
            logger.error(f"Error getting table info from SQLite: {e}")
            return {}
    
    def export_to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Export all data to pandas DataFrames for analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                dataframes = {}
                
                # Export each table
                for table in ['prompts', 'completions', 'votes', 'predictions', 'reflections']:
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    dataframes[table] = df
                
                # Create combined annotation view
                annotation_query = """
                    SELECT 
                        v.*,
                        p.enhanced_prompt,
                        p.domain,
                        p.difficulty,
                        ca.completion_text as completion_a_text,
                        cb.completion_text as completion_b_text
                    FROM votes v
                    JOIN prompts p ON v.prompt_id = p.id
                    JOIN completions ca ON v.completion_a_id = ca.id
                    JOIN completions cb ON v.completion_b_id = cb.id
                """
                dataframes['full_annotations'] = pd.read_sql_query(annotation_query, conn)
                
                return dataframes
                
        except Exception as e:
            logger.error(f"Error exporting to DataFrames: {e}")
            return {}

def migrate_from_jsonl(jsonl_data_dir: str, sqlite_db_path: str = None) -> bool:
    """Migrate existing JSONL data to SQLite"""
    try:
        # Initialize SQLite database
        db = RLHFSQLiteDB(sqlite_db_path)
        
        # Read existing JSONL files
        data_dir = Path(jsonl_data_dir)
        votes_file = data_dir / "votes.jsonl"
        
        if votes_file.exists():
            logger.info("Migrating votes.jsonl to SQLite...")
            
            with open(votes_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        vote_data = json.loads(line.strip())
                        
                        # Convert JSONL format to SQLite format
                        if 'completions' in vote_data and len(vote_data['completions']) >= 2:
                            annotation_data = {
                                "prompt_id": f"migrated_prompt_{line_num}",
                                "prompt": vote_data.get("prompt", ""),
                                "completion_a": vote_data['completions'][0],
                                "completion_b": vote_data['completions'][1],
                                "preference": "Completion A" if vote_data.get("chosen_index", 0) == 0 else "Completion B",
                                "confidence": vote_data.get("confidence", 0.8),
                                "feedback": vote_data.get("annotation", ""),
                                "model_provider": "migrated"
                            }
                            
                            db.save_annotation(annotation_data)
                            
                    except Exception as e:
                        logger.warning(f"Error migrating line {line_num}: {e}")
            
            logger.info(f"Successfully migrated data from {votes_file}")
            return True
        else:
            logger.warning(f"No votes.jsonl found at {votes_file}")
            return False
            
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return False

if __name__ == "__main__":
    # Test the database
    print("🗄️  Testing SQLite RLHF Database...")
    
    db = RLHFSQLiteDB()
    
    # Test annotation save
    test_annotation = {
        "prompt_id": "test_prompt_001",
        "prompt": "Explain the concept of recursion in programming.",
        "completion_a": "Recursion is when a function calls itself...",
        "completion_b": "A recursive function is one that calls itself...",
        "preference": "Completion A",
        "confidence": 0.85,
        "feedback": "More clear explanation",
        "quality_metrics": {"clarity": "high", "accuracy": "medium"}
    }
    
    success = db.save_annotation(test_annotation)
    print(f"✅ Test annotation saved: {success}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"📊 Database stats: {stats}")
    
    # Get annotations
    annotations = db.get_annotations(limit=5)
    print(f"📝 Retrieved {len(annotations)} annotations")
    
    print("🎯 SQLite RLHF Database ready for production!") 