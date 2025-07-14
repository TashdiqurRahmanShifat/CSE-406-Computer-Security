from flask import Flask, send_from_directory, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import json
import os
from datetime import datetime
import torch
import torch.nn as nn
from train import ComplexFingerprintClassifier, FingerprintClassifier


app = Flask(__name__)

stored_traces = []
# stored_heatmaps = []


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


#Received by Backend from Frontend
@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    """ 
    Implement the collect_trace endpoint to receive trace data from the frontend and generate a heatmap.
    1. Receive trace data from the frontend as JSON
    2. Generate a heatmap using matplotlib
    3. Store the heatmap and trace data in the backend temporarily
    4. Return the heatmap image and optionally other statistics to the frontend
    """
    try:
        # Receive trace data from the frontend as JSON
        data = request.get_json()
        trace_data = data.get('trace_data', [])#The default [] ensures the code doesn't crash if the key is missing
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        if not trace_data:
            return jsonify({'error': 'No trace data provided'}), 400
        
        # Convert trace data to numpy array for easier manipulation
        trace_array = np.array(trace_data)
        
        # Generate heatmap visualization to match reference style
        plt.figure(figsize=(12, 2))
        
        # Reshape data for heatmap (single row)
        heatmap_data = trace_array.reshape(1, -1)
        
        # Create heatmap with plasma colormap for purple-yellow bands
        im = plt.imshow(heatmap_data, cmap='plasma', aspect='auto', interpolation='none')
        
        # Customize the plot
        plt.title(f'Cache Sweep Count Trace - {timestamp[:19]}')
        plt.xlabel('Time Window')
        plt.ylabel('Trace')
        plt.colorbar(im, label='Sweep Count')
        
        # Remove y-axis ticks since we only have one row
        plt.yticks([])
        
        # Save plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()  # Close the figure to free memory
        
        # Create data URL for the image
        heatmap_url = f"data:image/png;base64,{img_base64}"
        
        # Calculate statistics(Sweep Count Trace)
        stats = {
            'min': int(np.min(trace_array)),
            'max': int(np.max(trace_array)),
            'mean': float(np.mean(trace_array)),
            'std': float(np.std(trace_array)),
            'range': int(np.max(trace_array) - np.min(trace_array)),
            'samples': len(trace_data)
        }
        
        # Store both the raw data and the visualization
        trace_entry = {
            'timestamp': timestamp,
            'trace_data': trace_data,# Raw data
            'heatmap_url': heatmap_url,# Visualization
            'stats': stats
        }
        
        stored_traces.append(trace_entry)
        # stored_heatmaps.append(heatmap_url)
        
        # Create info string
        info = f"Min: {stats['min']}, Max: {stats['max']}, Range: {stats['range']}, Samples: {stats['samples']}"
        
        # Return the image and data to display in the frontend
        return jsonify({
            'success': True,
            'heatmap_url': heatmap_url,
            'stats': stats,
            'info': info,
            'message': 'Trace processed successfully'
        })
        
    except Exception as e:
        print(f"Error in collect_trace: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_traces', methods=['GET'])
def download_traces():
    """
    Download all collected traces as JSON
    """
    try:
        # Create a clean version of traces without the base64 image data for download
        clean_traces = []
        for trace in stored_traces:
            clean_trace = {
                'timestamp': trace['timestamp'],
                'trace_data': trace['trace_data'],
                'stats': trace['stats']
            }
            clean_traces.append(clean_trace)
        
        # Convert to JSON string
        json_data = json.dumps(clean_traces, indent=2)
        
        # Create response
        response = app.response_class(
            response=json_data,
            status=200,
            mimetype='application/json'
        )
        response.headers['Content-Disposition'] = 'attachment; filename=traces.json'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """ 
    Implement a clear results endpoint to reset stored data.
    1. Clear stored traces and heatmaps
    2. Return success/error message
    """
    try:
        global stored_traces
        # , stored_heatmaps
        
        # Clear all stored data
        stored_traces.clear()
        # stored_heatmaps.clear()
        
        return jsonify({
            'success': True,
            'message': 'All results cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


MODEL = None
METADATA = None
SCALER_MEANS = None
SCALER_SCALES = None
IDX_TO_WEBSITE = None

def load_model_and_metadata():
    """
    Loads the trained model and metadata from disk.
    This function is called once when the application starts.
    """
    global MODEL, METADATA, SCALER_MEANS, SCALER_SCALES, IDX_TO_WEBSITE
    
    model_path = 'model.pth'
    metadata_path = 'model_metadata.json'
    
    print("Loading model and metadata...")
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print(f"Error: Model ('{model_path}') or metadata ('{metadata_path}') not found.")
        print("Please run train.py to generate these files.")
        return

    try:
        # Load metadata from JSON file
        with open(metadata_path, 'r') as f:
            METADATA = json.load(f)
        
        # Extract scaler parameters and website mapping
        scaler_params = METADATA['scaler_params']
        SCALER_MEANS = np.array(scaler_params['mean'])
        SCALER_SCALES = np.array(scaler_params['scale'])
        IDX_TO_WEBSITE = {int(k): v for k, v in METADATA['idx_to_website'].items()}

        # Determine which model architecture to use based on metadata
        input_size = METADATA['input_size']
        hidden_size = METADATA['hidden_size']
        num_classes = METADATA['num_classes']
        
        if METADATA['best_model'] == 'Complex CNN':
            MODEL = ComplexFingerprintClassifier(input_size, hidden_size, num_classes)
            print("Instantiated ComplexFingerprintClassifier.")
        else:
            MODEL = FingerprintClassifier(input_size, hidden_size, num_classes)
            print("Instantiated FingerprintClassifier.")
            
        # Load the trained model weights
        MODEL.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        MODEL.eval()  # Set the model to evaluation mode
        
        print("Model and metadata loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model and metadata: {e}")


@app.route('/predict', methods=['POST'])
def predict_website():
    """
    Real-time website prediction endpoint. This is the corrected version.
    """
    # Check if the model has been loaded successfully
    if not MODEL or not METADATA:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500
        
    try:
        data = request.get_json()
        trace_data = data.get('trace_data',[])
        
        if not trace_data:
            return jsonify({'error': 'No trace data provided'}), 400
        
        # Pad or truncate the trace to the required input size
        input_size = METADATA['input_size']
        if len(trace_data) < input_size:
            trace_mean = sum(trace_data) / len(trace_data) if trace_data else 0
            padded_trace = trace_data + [trace_mean] * (input_size - len(trace_data))
        else:
            padded_trace = trace_data[:input_size]

        padded_trace_np = np.array(padded_trace).reshape(1, -1) # Reshape for scaler

        # Apply the same StandardScaler normalization used during training
        normalized_trace = (padded_trace_np - SCALER_MEANS) / SCALER_SCALES
        
        # Convert to a PyTorch tensor
        trace_tensor = torch.tensor(normalized_trace, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            outputs = MODEL(trace_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence = torch.max(probabilities).item()
            predicted_idx = torch.argmax(probabilities, dim=1).item()
        
        # Map class index back to website name
        predicted_website = IDX_TO_WEBSITE.get(predicted_idx, f'Unknown (class {predicted_idx})')
        
        return jsonify({
            'success': True,
            'predicted_website': predicted_website,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        print(f"Error in predict_website: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model_and_metadata()
    app.run(debug=True, host='0.0.0.0', port=5000)