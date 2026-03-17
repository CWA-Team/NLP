"""
Model Manager for EquiLens Trained Models
Saves and loads trained bias detection models with optimized parameters
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import json


class ModelManager:
    """Manages saving and loading trained models."""
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to save trained models
        """
        if models_dir is None:
            # Default to src/data/models directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'data', 'models')
        
        self.models_dir = models_dir
        self._ensure_models_dir()
    
    def _ensure_models_dir(self):
        """Create models directory if it doesn't exist."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"Created models directory: {self.models_dir}")
    
    def save_model(self, model_name: str, parameters: Dict, metadata: Dict = None) -> str:
        """
        Save a trained model with its parameters.
        
        Args:
            model_name: Name for the model
            parameters: Model parameters (temperature, etc.)
            metadata: Additional metadata about the model
            
        Returns:
            Path to saved model file
        """
        # Create model data
        model_data = {
            'model_name': model_name,
            'parameters': parameters,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_{timestamp}.json"
        filepath = os.path.join(self.models_dir, filename)
        
        # Save model
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Also save as latest for easy access
        latest_path = os.path.join(self.models_dir, f"{model_name}_latest.json")
        with open(latest_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Saved model: {filepath}")
        return filepath
    
    def load_model(self, model_name: str = None, filepath: str = None) -> Optional[Dict]:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model to load
            filepath: Specific filepath to load from
            
        Returns:
            Model data dictionary or None
        """
        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        if model_name:
            # Try to load latest version
            latest_path = os.path.join(self.models_dir, f"{model_name}_latest.json")
            if os.path.exists(latest_path):
                with open(latest_path, 'r') as f:
                    return json.load(f)
        
        return None
    
    def list_models(self) -> List[Dict]:
        """
        List all available trained models.
        
        Returns:
            List of model info dictionaries
        """
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('_latest.json'):
                filepath = os.path.join(self.models_dir, filename)
                with open(filepath, 'r') as f:
                    model_data = json.load(f)
                    models.append({
                        'name': model_data.get('model_name'),
                        'filepath': filepath,
                        'created_at': model_data.get('created_at'),
                        'parameters': model_data.get('parameters', {}),
                        'metadata': model_data.get('metadata', {})
                    })
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return models
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful
        """
        # Delete latest version
        latest_path = os.path.join(self.models_dir, f"{model_name}_latest.json")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        
        # Delete all version files
        for filename in os.listdir(self.models_dir):
            if filename.startswith(f"{model_name}_") and filename.endswith('.json'):
                filepath = os.path.join(self.models_dir, filename)
                os.remove(filepath)
        
        return True


# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
