"""
Flask API for benchmarking RAG configurations with integrated evaluation system.
Handles configuration input, model execution, evaluation, and result reporting.
"""

import json
from typing import Dict, Any

# Flask imports
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# JWT Authentication
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, decode_token

# Benchmark components
from benchmarks import load_datasets_from_json

# RAG implementations
from graphrag import run_model as graphrag
from raptor import run_model as raptor
from rag import run_model as rag

# Evaluation system
from evaluations import evals
from consolidate_metrics import consolidation

# Initialize Flask application
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'mr0bunbustic@$3758'  # In production, use proper secret management
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})
jwt = JWTManager(app)

def generate_benchmarks(model_config: Dict, rag_config: Dict, benchmarks_data: Dict) -> Dict:
    """
    Route input to appropriate RAG implementation based on configuration.
    
    Args:
        model_config: Dictionary containing model parameters
        rag_config: Dictionary specifying RAG type and parameters
        benchmarks_data: Loaded benchmark datasets
        
    Returns:
        Dictionary of benchmark predictions
    """
    rag_type = rag_config.get('type', 'Vanilla RAG')
    
    if rag_type == 'Vanilla RAG':
        return rag(model_config, benchmarks_data)
    elif rag_type == 'RAPTOR':
        return raptor(model_config, benchmarks_data)
    elif rag_type == 'Graph RAG':
        return graphrag(model_config, benchmarks_data)
    else:
        raise ValueError(f"Unsupported RAG type: {rag_type}")

''' API Call to the frontend'''
@app.route('/benchmarks', methods=['POST'])
def benchmarks() -> Response:
    """
    Main benchmarking endpoint handling:
    - Configuration parsing
    - File upload processing
    - Model execution
    - Evaluation and result consolidation
    """
    try:
        # Parse main configuration from form data
        config = json.loads(request.form.get('config', '{}'))
        if not config:
            return jsonify({'error': 'Missing configuration data'}), 400

        # Process file uploads if present
        uploaded_files_data = {}
        if 'files' in request.files:
            for file in request.files.getlist('files'):
                if file.filename.endswith('.json') and file.filename != '':
                    file_content = json.loads(file.read().decode('utf-8'))
                    uploaded_files_data[file.filename[:-5]] = file_content  # Remove .json extension

        # Load configuration and benchmarks
        with open('config.json', 'r') as f:
            default_config = json.load(f)
        
        # Load standard benchmark datasets
        benchmark_datasets = load_datasets_from_json(default_config['BENCHMARKS'],default_config['Num_Samples'])

        # Execute core benchmarking
        model_config = config['MODEL_CONFIG']
        rag_config = config['RAG_CONFIG']
        predictions = generate_benchmarks(model_config, rag_config, benchmark_datasets)
        
        # Evaluate and consolidate results
        evaluation_results = evals(predictions)
        consolidated_metrics = consolidation(evaluation_results)

        # Process user-uploaded files if any
        if uploaded_files_data:
            custom_predictions = generate_benchmarks(model_config, rag_config, uploaded_files_data)
            custom_evaluation = evals(custom_predictions)
            custom_metrics = consolidation(custom_evaluation)
            
            # Merge standard and custom results
            final_metrics = {**consolidated_metrics, **custom_metrics}
            full_report = {**evaluation_results, **custom_evaluation}
        else:
            final_metrics = consolidated_metrics
            full_report = evaluation_results

        return jsonify({
            "results": final_metrics,
            "report": full_report
        })

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON configuration'}), 400
    except KeyError as e:
        return jsonify({'error': f'Missing configuration field: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)