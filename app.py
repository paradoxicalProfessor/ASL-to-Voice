"""
ASL to Voice - Web Application
Flask-based web interface for real-time ASL detection with webcam
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import json
import time
from collections import deque, Counter
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Global variables
model = None
detection_history = deque(maxlen=15)
current_word = ""
sentence = []
last_letter = None
last_letter_time = 0
letter_cooldown = 1.5
smoothing_window = 15
min_detection_frames = 8
conf_threshold = 0.6

def load_model():
    """Load YOLOv8 model"""
    global model
    if model is None:
        print("📦 Loading YOLOv8 model...")
        model = YOLO('runs/train/asl_detection/weights/best.pt')
        print("✅ Model loaded successfully")
    return model

def get_smoothed_prediction():
    """Get most common prediction over recent frames"""
    if len(detection_history) < min_detection_frames:
        return None, 0.0
    
    letter_counts = Counter(detection_history)
    
    if letter_counts:
        most_common_letter, count = letter_counts.most_common(1)[0]
        confidence = count / len(detection_history)
        
        if count >= min_detection_frames:
            return most_common_letter, confidence
    
    return None, 0.0

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Process frame and return detection results"""
    global detection_history
    
    try:
        # Get image from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Load model
        model = load_model()
        
        # Run detection
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Extract detections
        detected_letter = None
        confidence = 0.0
        bbox = None
        
        if len(results[0].boxes) > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            best_idx = np.argmax(confidences)
            detected_letter = model.names[int(classes[best_idx])]
            confidence = float(confidences[best_idx])
            bbox = boxes[best_idx].astype(int).tolist()
            
            detection_history.append(detected_letter)
        else:
            detection_history.append(None)
        
        # Get smoothed prediction
        smoothed_letter, smoothed_conf = get_smoothed_prediction()
        
        return jsonify({
            'success': True,
            'detected_letter': detected_letter,
            'confidence': confidence,
            'bbox': bbox,
            'smoothed_letter': smoothed_letter,
            'smoothed_confidence': float(smoothed_conf) if smoothed_conf else 0.0,
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/add_letter', methods=['POST'])
def add_letter():
    """Add detected letter to current word"""
    global current_word, last_letter, last_letter_time, detection_history
    
    try:
        data = request.get_json()
        letter = data['letter']
        
        current_time = time.time()
        
        # Check cooldown
        if letter == last_letter and (current_time - last_letter_time) < letter_cooldown:
            return jsonify({
                'success': False,
                'message': 'Cooldown period active'
            })
        
        current_word += letter
        last_letter = letter
        last_letter_time = current_time
        detection_history.clear()
        
        return jsonify({
            'success': True,
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/add_space', methods=['POST'])
def add_space():
    """Add space (finalize current word)"""
    global current_word, sentence, last_letter
    
    try:
        if current_word:
            sentence.append(current_word)
            current_word = ""
            last_letter = None
        
        return jsonify({
            'success': True,
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/add_punctuation', methods=['POST'])
def add_punctuation():
    """Add punctuation (period or comma)"""
    global current_word, sentence, last_letter
    
    try:
        data = request.get_json()
        punct = data['punctuation']
        
        if current_word:
            sentence.append(current_word)
            current_word = ""
        
        if sentence and punct in ['.', ',', '!', '?']:
            sentence[-1] += punct
            last_letter = None
        
        return jsonify({
            'success': True,
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/delete_char', methods=['POST'])
def delete_char():
    """Delete last character from current word"""
    global current_word
    
    try:
        if current_word:
            current_word = current_word[:-1]
        
        return jsonify({
            'success': True,
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/clear_word', methods=['POST'])
def clear_word():
    """Clear current word"""
    global current_word, last_letter
    
    try:
        current_word = ""
        last_letter = None
        
        return jsonify({
            'success': True,
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset entire sentence"""
    global current_word, sentence, last_letter
    
    try:
        current_word = ""
        sentence = []
        last_letter = None
        
        return jsonify({
            'success': True,
            'current_word': current_word,
            'sentence': ' '.join(sentence)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    """Get current sentence for TTS"""
    try:
        text = ' '.join(sentence)
        return jsonify({
            'success': True,
            'sentence': text
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ASL to Voice - Web Application")
    print("="*60)
    print("\n📦 Loading model...")
    load_model()
    print("\n✅ Server ready!")
    print("🌐 Open: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
