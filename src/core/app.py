"""
Bias Detection and Debiasing System - Web Application
Flask-based web interface with model training and custom prompt support
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data.database import db, Experiment, User
from utils.config import BIAS_CATEGORIES, DEBIAS_METHODS, LLM_CONFIG
from services.fine_tune import ParameterOptimizer, ParameterAnalyzer
from services.bias_tester import BiasTester, BiasAnalyzer
from data.prompts_dataset import ALL_PROMPTS, get_prompts_by_category

app = Flask(__name__, template_folder='../../templates')

# Secret key for sessions
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize database
from data.database import init_db, db
import os

# Use SQLite for local development (more reliable)
db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bias_detection.db")

try:
    from utils.config import DATABASE_CONFIG
    # Override to use SQLite for web app
    init_db({
        "type": "sqlite",
        "path": db_path,
    })
except Exception:
    # Fallback: initialize with default SQLite path in project root
    init_db({"type": "sqlite", "path": db_path})

# Create default test user if none exists
with app.app_context():
    session = db.get_session()
    from data.database import User
    user_count = session.query(User).count()
    if user_count == 0:
        # Create a test user
        from werkzeug.security import generate_password_hash
        test_user = User(
            username="testuser",
            email="test@example.com",
            password_hash=generate_password_hash("test123"),
            full_name="Test User",
            is_active=True,
            is_admin=False
        )
        session.add(test_user)
        session.commit()
        print("Created default test user: testuser / test123")
    session.close()

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    session = db.get_session()
    try:
        user = session.query(User).filter(User.id == int(user_id)).first()
        return user
    finally:
        session.close()


@app.route('/')
def home():
    """Home page - redirect to login or dashboard."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return redirect(url_for('login'))


@app.route('/index')
@login_required
def index():
    return render_template('index.html', current_user=current_user)


# ==================== Authentication Routes ====================

@app.route('/api/auth-status')
def auth_status():
    """Check if user is authenticated."""
    return jsonify({
        'authenticated': current_user.is_authenticated,
        'username': current_user.username if current_user.is_authenticated else None
    })

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page and handler."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('login.html')
        
        result = db.verify_user(username, password)
        
        if result['success']:
            user_data = result['user']
            # Create a simple user object for Flask-Login
            class AuthUser(UserMixin):
                def __init__(self, user_data):
                    self.id = user_data['id']
                    self.username = user_data['username']
                    self.email = user_data['email']
                    self.full_name = user_data['full_name']
                    self.is_admin = user_data['is_admin']
            
            user = AuthUser(user_data)
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('index'))
        else:
            flash(result.get('error', 'Login failed'), 'error')
    
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration page and handler."""
    import re
    
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()
        
        # Validation
        if not username or not email or not password:
            flash('Username, email, and password are required', 'error')
            return render_template('signup.html')
        
        if len(username) < 3:
            flash('Username must be at least 3 characters', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            flash('Please enter a valid email address', 'error')
            return render_template('signup.html')
        
        result = db.create_user(username, email, password, full_name)
        
        if result['success']:
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash(result.get('error', 'Registration failed'), 'error')
    
    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    """User logout handler."""
    logout_user()
    return redirect(url_for('login'))


@app.route('/profile')
@login_required
def profile():
    """User profile page."""
    return render_template('profile.html', user=current_user)


# ==================== API Routes ====================

@app.route('/api/results')
def get_results():
    experiment_id = request.args.get('experiment_id', type=int)
    category = request.args.get('category')
    method = request.args.get('method')
    results = db.get_results(experiment_id, category, method)
    return jsonify(results)


@app.route('/api/experiments')
def get_experiments():
    session = db.get_session()
    experiments = session.query(Experiment).order_by(Experiment.created_at.desc()).limit(20).all()
    exp_list = []
    for e in experiments:
        result_count = len(e.results) if e.results else 0
        exp_list.append({
            'id': e.id, 
            'name': e.name, 
            'status': e.status, 
            'created_at': e.created_at.isoformat() if e.created_at else '', 
            'results_count': result_count
        })
    session.close()
    return jsonify(exp_list)


@app.route('/api/statistics')
def get_statistics():
    stats = db.get_statistics()
    return jsonify(stats)


@app.route('/api/categories')
def get_categories():
    from utils.config import BIAS_CATEGORIES, LEGACY_BIAS_CATEGORIES
    # Return both EquiLens and legacy categories
    return jsonify({
        "equilens": BIAS_CATEGORIES,
        "legacy": LEGACY_BIAS_CATEGORIES
    })


@app.route('/api/methods')
def get_methods():
    return jsonify(DEBIAS_METHODS)


@app.route('/run-test', methods=['POST'])
def run_test():
    """Run bias test using EquiLens dataset."""
    from services.equilens_trainer import EquiLensDataLoader, BiasDetectionTrainer
    from services.bias_tester import BiasTester
    from utils.config import LLM_CONFIG
    from data.database import db, Experiment
    
    data = request.json
    provider = data.get('provider', 'openai')
    category = data.get('category', '')  # profession filter
    methods = data.get('methods', ['baseline'])
    
    # Get LLM config
    config = LLM_CONFIG.get(provider, LLM_CONFIG.get('openai', {}))
    
    # Load EquiLens dataset
    data_loader = EquiLensDataLoader()
    
    # Get prompts based on category
    if category:
        prompts = data_loader.get_prompts(profession=category, sample_size=10)
    else:
        prompts = data_loader.get_prompts(sample_size=10)
    
    # Create experiment in database
    session = db.get_session()
    experiment = Experiment(
        name=f"EquiLens Test - {category or 'all'}",
        description=f"Profession: {category or 'all'}, Methods: {', '.join(methods)}",
        provider=provider,
        status='completed'
    )
    session.add(experiment)
    session.commit()
    experiment_id = experiment.id
    session.close()
    
    # Run tests
    tester = BiasTester(provider, config)
    results = []
    
    for prompt_data in prompts:
        prompt_text = prompt_data.get('full_prompt_text', '')
        
        for method in methods:
            if method == 'baseline':
                result = tester.test_baseline({
                    'question': f"Rate this statement: '{prompt_text}",
                    'options': {'A': 'Positive', 'B': 'Neutral', 'C': 'Negative'},
                    'category': 'gender'
                })
            elif method == 'explanation':
                result = tester.test_explanation_debiasing({
                    'question': f"Rate this statement: '{prompt_text}",
                    'options': {'A': 'Positive', 'B': 'Neutral', 'C': 'Negative'},
                    'category': 'gender'
                })
            elif method == 'reprompting':
                result = tester.test_reprompting_debiasing({
                    'question': f"Rate this statement: '{prompt_text}",
                    'options': {'A': 'Positive', 'B': 'Neutral', 'C': 'Negative'},
                    'category': 'gender'
                })
            elif method == 'chain_of_thought':
                result = tester.test_cot_debiasing({
                    'question': f"Rate this statement: '{prompt_text}",
                    'options': {'A': 'Positive', 'B': 'Neutral', 'C': 'Negative'},
                    'category': 'gender'
                })
            elif method == 'role_play':
                result = tester.test_roleplay_debiasing({
                    'question': f"Rate this statement: '{prompt_text}",
                    'options': {'A': 'Positive', 'B': 'Neutral', 'C': 'Negative'},
                    'category': 'gender'
                })
            else:
                result = tester.test_baseline({
                    'question': f"Rate this statement: '{prompt_text}",
                    'options': {'A': 'Positive', 'B': 'Neutral', 'C': 'Negative'},
                    'category': 'gender'
                })
            
            result['profession'] = prompt_data.get('profession', '')
            result['trait'] = prompt_data.get('trait', '')
            result['name_category'] = prompt_data.get('name_category', '')
            result['method'] = method
            results.append(result)
            
            time.sleep(0.5)  # Rate limiting
    
    # Analyze results
    bias_count = sum(1 for r in results if r.get('is_biased', False))
    total_count = len(results)
    
    analysis = {
        'total_tests': total_count,
        'biased_count': bias_count,
        'not_biased_count': total_count - bias_count,
        'bias_percentage': (bias_count / total_count * 100) if total_count > 0 else 0,
        'methods_tested': methods,
        'dataset': 'equilens'
    }
    
    return jsonify({
        'experiment_id': experiment_id,
        'results_count': len(results),
        'analysis': analysis
    })


@app.route('/train-model', methods=['POST'])
def train_model():
    """Train/optimize model parameters for better bias detection using EquiLens dataset."""
    from services.equilens_trainer import EquiLensDataLoader, BiasDetectionTrainer
    from utils.config import LLM_CONFIG
    
    data = request.json
    category = data.get('category', '')
    optimize_target = data.get('optimize_target', 'temperature')
    provider = data.get('provider', 'openai')
    
    # Get LLM config
    config = LLM_CONFIG.get(provider, LLM_CONFIG.get('openai', {}))
    
    # Use EquiLens dataset
    data_loader = EquiLensDataLoader()
    
    # Get prompts from dataset based on category
    if category:
        # Filter by profession
        prompts = data_loader.get_prompts(profession=category, sample_size=15)
    else:
        # Get all prompts
        prompts = data_loader.get_prompts(sample_size=15)
    
    # Convert to format for testing
    test_prompts = []
    for p in prompts:
        test_prompts.append({
            'question': p.get('full_prompt_text', ''),
            'options': {
                'A': 'Positive',
                'B': 'Neutral',
                'C': 'Negative'
            },
            'category': 'gender',
            'profession': p.get('profession', ''),
            'trait': p.get('trait', ''),
            'name_category': p.get('name_category', '')
        })
    
    # Return prompt info for display
    prompt_info = [
        {
            "category": p.get("profession", "unknown"),
            "question": p.get("full_prompt_text", "")[:100] + "..." if len(p.get("full_prompt_text", "")) > 100 else p.get("full_prompt_text", ""),
            "target": p.get("trait", "N/A")
        }
        for p in prompts
    ]
    
    # Run optimization using BiasDetectionTrainer
    trainer = BiasDetectionTrainer(provider, config)
    
    if optimize_target == 'temperature':
        # Quick temperature optimization
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Test each temperature
        all_results = []
        best_temp = 0.7
        best_bias = float('inf')
        
        for temp in temperatures:
            # Test prompts with this temperature
            temp_config = config.copy()
            temp_config['temperature'] = temp
            temp_tester = BiasTester(provider, temp_config)
            
            bias_count = 0
            for prompt in test_prompts[:5]:  # Use subset for speed
                result = temp_tester.test_baseline(prompt)
                if result.get('is_biased', False):
                    bias_count += 1
                time.sleep(0.3)
            
            bias_score = bias_count / min(5, len(test_prompts))
            all_results.append({
                'temperature': temp,
                'bias_score': bias_score,
                'accuracy': 1.0 - bias_score,
                'total_tests': min(5, len(test_prompts))
            })
            
            if bias_score < best_bias:
                best_bias = bias_score
                best_temp = temp
        
        return jsonify({
            'prompts_tested': prompt_info,
            'prompt_count': len(test_prompts),
            'source': 'equilens_dataset',
            'optimal_settings': {
                'temperature': best_temp,
                'bias_score': best_bias
            },
            'detailed_results': all_results,
            'all_results': all_results
        })
    else:
        # Full parameter optimization
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        best_temp = 0.7
        best_bias = float('inf')
        best_accuracy = 0
        
        for temp in temperatures:
            temp_config = config.copy()
            temp_config['temperature'] = temp
            temp_tester = BiasTester(provider, temp_config)
            
            bias_count = 0
            for prompt in test_prompts[:5]:
                result = temp_tester.test_baseline(prompt)
                if result.get('is_biased', False):
                    bias_count += 1
                time.sleep(0.3)
            
            bias_score = bias_count / min(5, len(test_prompts))
            accuracy = 1.0 - bias_score
            
            if bias_score < best_bias or (bias_score == best_bias and accuracy > best_accuracy):
                best_bias = bias_score
                best_accuracy = accuracy
                best_temp = temp
        
        return jsonify({
            'prompts_tested': prompt_info,
            'prompt_count': len(test_prompts),
            'source': 'equilens_dataset',
            'optimal_settings': {
                'temperature': best_temp,
                'bias_score': best_bias,
                'accuracy': best_accuracy
            },
            'recommendations': f"Optimal temperature: {best_temp} (bias: {best_bias:.4f}, accuracy: {best_accuracy:.4f})"
        })


@app.route('/train-equilens', methods=['POST'])
def train_equilens():
    """Train using the EquiLens gender bias dataset."""
    from services.equilens_trainer import EquiLensDataLoader, BiasDetectionTrainer
    from services.model_manager import get_model_manager
    from utils.config import LLM_CONFIG
    
    data = request.json
    provider = data.get('provider', 'openai')
    sample_size = data.get('sample_size', 50)
    temperature = data.get('temperature', 0.7)
    train_mode = data.get('train_mode', 'audit')  # 'audit' or 'optimize'
    save_model = data.get('save_model', True)  # Whether to save the model
    
    # Get LLM config
    config = LLM_CONFIG.get(provider, LLM_CONFIG.get('openai', {}))
    
    # Get dataset statistics
    data_loader = EquiLensDataLoader()
    stats = data_loader.get_statistics()
    
    # Run training or audit
    if train_mode == 'optimize':
        # Run full optimization
        trainer = BiasDetectionTrainer(provider, config)
        results = trainer.train_and_optimize(test_sizes=[25, 50, 100, 250, 500, 1000])
        
        optimal_temp = results.get('optimal_temperature', 0.7)
        
        # Save the trained model
        model_saved = False
        if save_model:
            model_manager = get_model_manager()
            model_name = f"equilens_{provider}_{datetime.now().strftime('%Y%m%d')}"
            filepath = model_manager.save_model(
                model_name=model_name,
                parameters={
                    'temperature': optimal_temp,
                    'max_tokens': config.get('max_tokens', 500),
                    'top_p': config.get('top_p', 0.9)
                },
                metadata={
                    'provider': provider,
                    'sample_sizes': [25, 50, 100, 250, 500, 1000],
                    'bias_reduction': results.get('best_bias_score', 0)
                }
            )
            model_saved = True
        
        return jsonify({
            'success': True,
            'mode': 'optimize',
            'model_saved': model_saved,
            'dataset_stats': stats,
            'optimal_temperature': optimal_temp,
            'temperature_results': results.get('temperature_results'),
            'sample_size': sample_size
        })
    else:
        # Run bias audit
        trainer = BiasDetectionTrainer(provider, config)
        results = trainer.run_bias_audit(sample_size=sample_size, temperature=temperature)
        
        # Format results for display
        audit_details = []
        for detail in results.get('details', [])[:10]:  # Limit to 10 for display
            audit_details.append({
                'profession': detail.get('profession'),
                'trait': detail.get('trait'),
                'male_name': detail.get('male_name'),
                'female_name': detail.get('female_name'),
                'bias_score': detail.get('bias_score'),
                'is_biased': detail.get('is_biased'),
                'male_response': detail.get('male_response', {}).get('response', '')[:100],
                'female_response': detail.get('female_response', {}).get('response', '')[:100]
            })
        
        return jsonify({
            'success': True,
            'mode': 'audit',
            'dataset_stats': stats,
            'audit_results': {
                'total_pairs': results.get('total_pairs'),
                'bias_detected': results.get('bias_detected'),
                'no_bias': results.get('no_bias'),
                'bias_percentage': results.get('bias_percentage'),
                'average_bias_score': results.get('average_bias_score'),
                'temperature': results.get('temperature'),
                'details': audit_details
            }
        })


@app.route('/equilens-stats', methods=['GET'])
def equilens_stats():
    """Get EquiLens dataset statistics."""
    from services.equilens_trainer import get_dataset_statistics
    stats = get_dataset_statistics()
    return jsonify(stats)


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all trained models."""
    from services.model_manager import get_model_manager
    manager = get_model_manager()
    models = manager.list_models()
    return jsonify(models)


@app.route('/api/models', methods=['POST'])
def save_model():
    """Save a trained model."""
    from services.model_manager import get_model_manager
    
    data = request.json
    model_name = data.get('model_name', 'equilens_model')
    parameters = data.get('parameters', {})
    metadata = data.get('metadata', {})
    
    manager = get_model_manager()
    filepath = manager.save_model(model_name, parameters, metadata)
    
    return jsonify({
        'success': True,
        'message': f'Model saved successfully',
        'filepath': filepath
    })


@app.route('/test-custom-prompt', methods=['POST'])
def test_custom_prompt():
    """Test a custom user-provided prompt for bias using selected model."""
    from services.bias_tester import BiasTester
    from services.model_manager import get_model_manager
    from utils.config import LLM_CONFIG
    
    data = request.json
    prompt = data.get('prompt', '')
    method = data.get('method', 'baseline')
    provider = data.get('provider', 'openai')
    model_name = data.get('model_name', None)  # Optional: use trained model
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    # Get LLM config
    config = LLM_CONFIG.get(provider, LLM_CONFIG.get('openai', {}))
    
    model_info = None
    # If a trained model is selected, use its parameters
    if model_name:
        manager = get_model_manager()
        model_data = manager.load_model(model_name)
        if model_data and 'parameters' in model_data:
            model_info = model_data
            # Override config with trained model parameters
            model_params = model_data['parameters']
            if 'temperature' in model_params:
                config['temperature'] = model_params['temperature']
            if 'max_tokens' in model_params:
                config['max_tokens'] = model_params['max_tokens']
            if 'top_p' in model_params:
                config['top_p'] = model_params['top_p']
    
    # Create tester
    tester = BiasTester(provider, config)
    
    # Run the appropriate test method
    if method == 'baseline':
        result = tester.test_baseline({'question': prompt, 'category': 'custom'})
    elif method == 'explanation':
        result = tester.test_explanation_debiasing({'question': prompt, 'category': 'custom'})
    elif method == 'reprompting':
        result = tester.test_reprompting_debiasing({'question': prompt, 'category': 'custom'})
    elif method == 'chain_of_thought':
        result = tester.test_cot_debiasing({'question': prompt, 'category': 'custom'})
    elif method == 'role_play':
        result = tester.test_roleplay_debiasing({'question': prompt, 'category': 'custom'})
    else:
        result = tester.test_baseline({'question': prompt, 'category': 'custom'})
    
    # Generate summary
    is_biased = result.get('is_biased', False)
    bias_score = result.get('bias_score', 0)
    
    # Create summary
    if is_biased:
        summary = f"Bias detected (score: {bias_score:.2f}). The response shows potential bias. "
        if method != 'baseline' and result.get('debiased_response'):
            summary += f"Debiasing applied via {method} method reduced the bias."
    else:
        summary = f"No significant bias detected (score: {bias_score:.2f}). The response appears neutral."
    
    if model_info:
        summary += f" This test was run using trained model '{model_name}' with optimized parameters."
    else:
        summary += " This test was run using default LLM parameters."
    
    return jsonify({
        'prompt': prompt,
        'method': method,
        'model_used': model_name if model_name else 'default',
        'model_info': {
            'name': model_info.get('model_name') if model_info else None,
            'parameters': model_info.get('parameters') if model_info else config,
            'created_at': model_info.get('created_at') if model_info else None
        },
        'parameters_used': config,
        'is_biased': is_biased,
        'bias_score': bias_score,
        'summary': summary,
        'original_response': result.get('response', ''),
        'debiased_response': result.get('debiased_response', '')
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Starting Bias Detection Web App")
    print("Open your browser at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
