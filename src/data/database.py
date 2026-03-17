"""
Database Module for Bias Detection and Debiasing System
Handles all database operations using SQLAlchemy (supports SQLite and PostgreSQL)
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
import json

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, ForeignKey, TIMESTAMP, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import func, and_


Base = declarative_base()


class User(Base):
    """User model for authentication."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    last_login = Column(TIMESTAMP)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="user")
    
    # Flask-Login required properties
    @property
    def is_authenticated(self):
        return True
    
    @property
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)


class Experiment(Base):
    """Experiment model."""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)  # Link to user
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    status = Column(String, default='pending')
    config = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="experiments")
    results = relationship("Result", back_populates="experiment")
    summaries = relationship("Summary", back_populates="experiment")
    fine_tune_results = relationship("FineTuneResult", back_populates="experiment")


class Result(Base):
    """Bias test result model."""
    __tablename__ = 'results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    bias_category = Column(String, nullable=False)
    group_name = Column(String, nullable=False)
    method = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text)
    answer = Column(String)
    is_biased = Column(Boolean)
    bias_score = Column(Float)
    accuracy = Column(Float)
    response_time = Column(Float)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    experiment = relationship("Experiment", back_populates="results")


class Prompt(Base):
    """Prompt template model."""
    __tablename__ = 'prompts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String, nullable=False)
    group_name = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    options = Column(Text, nullable=False)
    correct_answer = Column(String)
    stereotype_target = Column(String)
    method = Column(String)
    baseline_response = Column(Text)
    debiased_response = Column(Text)
    explanation = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class Summary(Base):
    """AI-generated summary model."""
    __tablename__ = 'summaries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    summary_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metrics = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    experiment = relationship("Experiment", back_populates="summaries")


class FineTuneResult(Base):
    """Fine-tuning parameter test result model."""
    __tablename__ = 'fine_tune_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    parameter_name = Column(String, nullable=False)
    parameter_value = Column(Float, nullable=False)
    bias_score = Column(Float)
    accuracy = Column(Float)
    response_time = Column(Float)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    experiment = relationship("Experiment", back_populates="fine_tune_results")


class DatabaseManager:
    """Manages all database operations for the bias detection system."""
    
    def __init__(self, db_config: Dict = None):
        """
        Initialize database manager.
        
        Args:
            db_config: Database configuration dictionary with keys:
                - type: 'sqlite' or 'postgresql'
                - For SQLite: 'path' (database file path)
                - For PostgreSQL: 'host', 'port', 'database', 'user', 'password'
        """
        if db_config is None:
            db_config = {"type": "sqlite", "path": "bias_detection.db"}
        
        self.db_config = db_config
        self.engine = self._create_engine()
        self.Session = sessionmaker(bind=self.engine)
        self.init_database()
    
    def _create_engine(self):
        """Create database engine based on configuration."""
        db_type = self.db_config.get("type", "sqlite")
        
        if db_type == "postgresql":
            # PostgreSQL connection
            host = self.db_config.get("host", "localhost")
            port = self.db_config.get("port", 5432)
            database = self.db_config.get("database", "bias_detection")
            user = self.db_config.get("user", "postgres")
            password = self.db_config.get("password", "")
            
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            # SQLite connection (default)
            db_path = self.db_config.get("path", "bias_detection.db")
            connection_string = f"sqlite:///{db_path}"
        
        return create_engine(connection_string, echo=False)
    
    def init_database(self):
        """Initialize database tables."""
        Base.metadata.create_all(self.engine)
        
        # Migration: Add missing columns to existing tables
        self._migrate_schema()
    
    def _migrate_schema(self):
        """Add missing columns to existing tables (simple migration)."""
        import sqlite3
        import os
        
        # Only works for SQLite
        db_path = self.db_config.get('path')
        if self.db_config.get('type') != 'sqlite' or not db_path:
            return
        
        if not os.path.exists(db_path):
            return
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if experiments table exists and has user_id column
            cursor.execute("PRAGMA table_info(experiments)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'experiments' in [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
                if 'user_id' not in columns:
                    print("Migrating experiments table: adding user_id column...")
                    cursor.execute("ALTER TABLE experiments ADD COLUMN user_id INTEGER REFERENCES users(id)")
                    conn.commit()
                    print("Migration complete.")
            
            conn.close()
        except Exception as e:
            print(f"Migration note: {e}")
    
    def get_session(self):
        """Get a new database session."""
        return self.Session()
    
    def create_experiment(self, name: str, description: str, config: Dict) -> int:
        """Create a new experiment."""
        session = self.get_session()
        try:
            experiment = Experiment(
                name=name,
                description=description,
                config=json.dumps(config),
                status='pending'
            )
            session.add(experiment)
            session.commit()
            experiment_id = experiment.id
            return experiment_id
        finally:
            session.close()
    
    def update_experiment_status(self, experiment_id: int, status: str):
        """Update experiment status."""
        session = self.get_session()
        try:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                experiment.status = status
                session.commit()
        finally:
            session.close()
    
    def save_result(self, experiment_id: int, result: Dict) -> int:
        """Save a bias test result."""
        session = self.get_session()
        try:
            new_result = Result(
                experiment_id=experiment_id,
                bias_category=result.get('bias_category'),
                group_name=result.get('group_name'),
                method=result.get('method'),
                prompt=result.get('prompt'),
                response=result.get('response'),
                answer=result.get('answer'),
                is_biased=result.get('is_biased'),
                bias_score=result.get('bias_score'),
                accuracy=result.get('accuracy'),
                response_time=result.get('response_time')
            )
            session.add(new_result)
            session.commit()
            result_id = new_result.id
            return result_id
        finally:
            session.close()
    
    def save_prompt(self, prompt_data: Dict) -> int:
        """Save a prompt with its results."""
        session = self.get_session()
        try:
            prompt = Prompt(
                category=prompt_data.get('category'),
                group_name=prompt_data.get('group_name'),
                question=prompt_data.get('question'),
                options=json.dumps(prompt_data.get('options')),
                correct_answer=prompt_data.get('correct_answer'),
                stereotype_target=prompt_data.get('stereotype_target'),
                method=prompt_data.get('method'),
                baseline_response=prompt_data.get('baseline_response'),
                debiased_response=prompt_data.get('debiased_response'),
                explanation=prompt_data.get('explanation')
            )
            session.add(prompt)
            session.commit()
            prompt_id = prompt.id
            return prompt_id
        finally:
            session.close()
    
    def save_summary(self, experiment_id: int, summary_type: str, content: str, metrics: Dict = None) -> int:
        """Save an AI-generated summary."""
        session = self.get_session()
        try:
            summary = Summary(
                experiment_id=experiment_id,
                summary_type=summary_type,
                content=content,
                metrics=json.dumps(metrics) if metrics else None
            )
            session.add(summary)
            session.commit()
            summary_id = summary.id
            return summary_id
        finally:
            session.close()
    
    def save_fine_tune_result(self, experiment_id: int, param_name: str, param_value: float, 
                              bias_score: float, accuracy: float, response_time: float) -> int:
        """Save fine-tuning parameter test result."""
        session = self.get_session()
        try:
            ft_result = FineTuneResult(
                experiment_id=experiment_id,
                parameter_name=param_name,
                parameter_value=param_value,
                bias_score=bias_score,
                accuracy=accuracy,
                response_time=response_time
            )
            session.add(ft_result)
            session.commit()
            result_id = ft_result.id
            return result_id
        finally:
            session.close()
    
    def get_results(self, experiment_id: int = None, bias_category: str = None, 
                   method: str = None) -> List[Dict]:
        """Get results with optional filters."""
        session = self.get_session()
        try:
            query = session.query(Result)
            
            if experiment_id:
                query = query.filter(Result.experiment_id == experiment_id)
            if bias_category:
                query = query.filter(Result.bias_category == bias_category)
            if method:
                query = query.filter(Result.method == method)
            
            results = query.all()
            return [self._result_to_dict(r) for r in results]
        finally:
            session.close()
    
    def _result_to_dict(self, result: Result) -> Dict:
        """Convert Result object to dictionary."""
        return {
            'id': result.id,
            'experiment_id': result.experiment_id,
            'bias_category': result.bias_category,
            'group_name': result.group_name,
            'method': result.method,
            'prompt': result.prompt,
            'response': result.response,
            'answer': result.answer,
            'is_biased': result.is_biased,
            'bias_score': result.bias_score,
            'accuracy': result.accuracy,
            'response_time': result.response_time,
            'created_at': result.created_at.isoformat() if result.created_at else None
        }
    
    def get_prompts(self, category: str = None) -> List[Dict]:
        """Get prompts with optional filter."""
        session = self.get_session()
        try:
            query = session.query(Prompt)
            
            if category:
                query = query.filter(Prompt.category == category)
            
            prompts = query.all()
            return [self._prompt_to_dict(p) for p in prompts]
        finally:
            session.close()
    
    def _prompt_to_dict(self, prompt: Prompt) -> Dict:
        """Convert Prompt object to dictionary."""
        return {
            'id': prompt.id,
            'category': prompt.category,
            'group_name': prompt.group_name,
            'question': prompt.question,
            'options': json.loads(prompt.options) if prompt.options else [],
            'correct_answer': prompt.correct_answer,
            'stereotype_target': prompt.stereotype_target,
            'method': prompt.method,
            'baseline_response': prompt.baseline_response,
            'debiased_response': prompt.debiased_response,
            'explanation': prompt.explanation,
            'created_at': prompt.created_at.isoformat() if prompt.created_at else None
        }
    
    def get_summaries(self, experiment_id: int) -> List[Dict]:
        """Get all summaries for an experiment."""
        session = self.get_session()
        try:
            summaries = session.query(Summary).filter(
                Summary.experiment_id == experiment_id
            ).order_by(Summary.created_at).all()
            
            return [self._summary_to_dict(s) for s in summaries]
        finally:
            session.close()
    
    def _summary_to_dict(self, summary: Summary) -> Dict:
        """Convert Summary object to dictionary."""
        return {
            'id': summary.id,
            'experiment_id': summary.experiment_id,
            'summary_type': summary.summary_type,
            'content': summary.content,
            'metrics': json.loads(summary.metrics) if summary.metrics else None,
            'created_at': summary.created_at.isoformat() if summary.created_at else None
        }
    
    def get_fine_tune_results(self, experiment_id: int) -> List[Dict]:
        """Get fine-tuning results for an experiment."""
        session = self.get_session()
        try:
            results = session.query(FineTuneResult).filter(
                FineTuneResult.experiment_id == experiment_id
            ).order_by(FineTuneResult.parameter_name, FineTuneResult.parameter_value).all()
            
            return [self._ft_result_to_dict(r) for r in results]
        finally:
            session.close()
    
    def _ft_result_to_dict(self, result: FineTuneResult) -> Dict:
        """Convert FineTuneResult object to dictionary."""
        return {
            'id': result.id,
            'experiment_id': result.experiment_id,
            'parameter_name': result.parameter_name,
            'parameter_value': result.parameter_value,
            'bias_score': result.bias_score,
            'accuracy': result.accuracy,
            'response_time': result.response_time,
            'created_at': result.created_at.isoformat() if result.created_at else None
        }
    
    def get_statistics(self, experiment_id: int = None) -> Dict:
        """Get overall statistics."""
        session = self.get_session()
        try:
            query = session.query(
                Result.bias_category,
                Result.method,
                func.count(Result.id).label('count'),
                func.avg(Result.bias_score).label('avg_bias'),
                func.avg(Result.accuracy).label('avg_accuracy'),
                func.avg(Result.response_time).label('avg_time')
            ).group_by(Result.bias_category, Result.method)
            
            if experiment_id:
                query = query.filter(Result.experiment_id == experiment_id)
            
            rows = query.all()
            
            stats = {}
            for row in rows:
                category = row[0]
                if category not in stats:
                    stats[category] = {}
                stats[category][row[1]] = {
                    'count': row[2],
                    'avg_bias': float(row[3]) if row[3] else None,
                    'avg_accuracy': float(row[4]) if row[4] else None,
                    'avg_time': float(row[5]) if row[5] else None
                }
            
            return stats
        finally:
            session.close()
    
    def export_results_csv(self, filepath: str, experiment_id: int = None):
        """Export results to CSV."""
        import csv
        
        session = self.get_session()
        try:
            query = session.query(Result)
            
            if experiment_id:
                query = query.filter(Result.experiment_id == experiment_id)
            
            results = query.all()
            
            fieldnames = ['id', 'experiment_id', 'bias_category', 'group_name', 'method',
                         'prompt', 'response', 'answer', 'is_biased', 'bias_score', 
                         'accuracy', 'response_time', 'created_at']
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(self._result_to_dict(row))
        finally:
            session.close()
    
    # ==================== User Authentication Methods ====================
    
    def create_user(self, username: str, email: str, password: str, full_name: str = None) -> Dict:
        """Create a new user with hashed password."""
        from werkzeug.security import generate_password_hash
        
        session = self.get_session()
        try:
            # Check if username or email already exists
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                if existing_user.username == username:
                    return {'success': False, 'error': 'Username already exists'}
                else:
                    return {'success': False, 'error': 'Email already exists'}
            
            # Create new user with hashed password
            password_hash = generate_password_hash(password)
            new_user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                full_name=full_name,
                is_active=True,
                is_admin=False
            )
            session.add(new_user)
            session.commit()
            
            return {
                'success': True,
                'user': self._user_to_dict(new_user)
            }
        except Exception as e:
            session.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    def verify_user(self, username: str, password: str) -> Dict:
        """Verify user credentials and return user info if valid."""
        from werkzeug.security import check_password_hash
        
        session = self.get_session()
        try:
            user = session.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            if not user.is_active:
                return {'success': False, 'error': 'Account is deactivated'}
            
            if not check_password_hash(user.password_hash, password):
                return {'success': False, 'error': 'Invalid password'}
            
            # Update last login
            user.last_login = datetime.utcnow()
            session.commit()
            
            return {
                'success': True,
                'user': self._user_to_dict(user)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    def get_user_by_id(self, user_id: int) -> Dict:
        """Get user by ID."""
        session = self.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                return {'success': True, 'user': self._user_to_dict(user)}
            return {'success': False, 'error': 'User not found'}
        finally:
            session.close()
    
    def get_user_by_username(self, username: str) -> Dict:
        """Get user by username."""
        session = self.get_session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if user:
                return {'success': True, 'user': self._user_to_dict(user)}
            return {'success': False, 'error': 'User not found'}
        finally:
            session.close()
    
    def update_user_password(self, user_id: int, new_password: str) -> Dict:
        """Update user password."""
        from werkzeug.security import generate_password_hash
        
        session = self.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            user.password_hash = generate_password_hash(new_password)
            session.commit()
            
            return {'success': True}
        except Exception as e:
            session.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    def delete_user(self, user_id: int) -> Dict:
        """Deactivate a user account."""
        session = self.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            user.is_active = False
            session.commit()
            
            return {'success': True}
        except Exception as e:
            session.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            session.close()
    
    def _user_to_dict(self, user: User) -> Dict:
        """Convert User object to dictionary."""
        return {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
            'is_active': user.is_active,
            'is_admin': user.is_admin,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'last_login': user.last_login.isoformat() if user.last_login else None
        }


# Singleton instance - will be initialized with config
db = None


def init_db(config: Dict = None):
    """Initialize the database with configuration."""
    global db
    db = DatabaseManager(config)
    return db


# Initialize with default config for backward compatibility
# Only initialize if not already initialized (avoid import errors)
if db is None:
    try:
        from config import DATABASE_CONFIG
        db = DatabaseManager({
            "type": DATABASE_CONFIG.get("type", "sqlite"),
            "path": DATABASE_CONFIG.get("path", "bias_detection.db"),
            "host": DATABASE_CONFIG.get("host", "localhost"),
            "port": DATABASE_CONFIG.get("port", 5432),
            "database": DATABASE_CONFIG.get("database", "bias_detection"),
            "user": DATABASE_CONFIG.get("user", "postgres"),
            "password": DATABASE_CONFIG.get("password", "")
        })
    except (ImportError, AttributeError):
        # Fallback to SQLite if config not available
        db = DatabaseManager({"type": "sqlite", "path": "bias_detection.db"})
