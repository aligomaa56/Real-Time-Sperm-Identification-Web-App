import os
import time
import base64
import io
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
import numpy as np
import cv2
from .detection import SpermDetector
from .utils import preprocess_image, draw_annotations

app = Flask(__name__)
app.config['SECRET_KEY'] = 'spermsecret2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the detector
detector = SpermDetector()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('screen_capture')
def handle_screen_capture(data):
    try:
        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to numpy array for OpenCV processing
        img_np = np.array(img)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Preprocess the image
        preprocessed_img = preprocess_image(img_rgb)
        
        # Detect sperm cells
        results = detector.detect_sperm(preprocessed_img)
        
        # Draw annotations on the original image
        annotated_img = draw_annotations(img_np, results)
        
        # Convert back to base64 for sending to the client
        _, buffer = cv2.imencode('.png', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send the results back to the client
        emit('detection_results', {
            'image': f'data:image/png;base64,{img_base64}',
            'stats': {
                'live': results['live_count'],
                'dead': results['dead_count'],
                'abnormal': results['abnormal_count'],
                'total': results['total_count'],
                'live_percentage': results['live_percentage'],
                'dead_percentage': results['dead_percentage'],
                'abnormal_percentage': results['abnormal_percentage']
            }
        })
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)