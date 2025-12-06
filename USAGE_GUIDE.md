# ASL to Voice - Complete Usage Guide

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Training Pipeline](#training-pipeline)
3. [Model Evaluation](#model-evaluation)
4. [Model Export](#model-export)
5. [Live Inference](#live-inference)
6. [Batch Testing](#batch-testing)
7. [Advanced Configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)

---

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)

```powershell
# Run installation script
.\install.ps1

# Then run interactive setup
python quick_start.py
```

### Option 2: Manual Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA (for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🎓 Training Pipeline

### Basic Training

```python
# Using default settings (recommended)
python train_yolov8.py
```

This will:
- Use YOLOv8-small model
- Train for 150 epochs
- Use batch size 16
- Enable GPU automatically
- Save best model to `runs/train/asl_detection/weights/best.pt`

### Custom Training

```python
from train_yolov8 import train_yolov8_asl

# Fast training (for testing)
train_yolov8_asl(
    model_size='n',      # Nano model (fastest)
    epochs=50,
    batch_size=16,
    imgsz=640,
)

# High accuracy training
train_yolov8_asl(
    model_size='m',      # Medium model
    epochs=200,
    batch_size=8,        # Reduce if GPU memory issues
    imgsz=800,           # Higher resolution
    patience=40,
)
```

### Training Parameters Explained

| Parameter | Options | Description |
|-----------|---------|-------------|
| `model_size` | n, s, m, l, x | Model size (nano to xlarge) |
| `epochs` | 50-300 | Training iterations |
| `batch_size` | 4-32 | Images per batch |
| `imgsz` | 416-1280 | Input image size |
| `patience` | 20-50 | Early stopping patience |
| `device` | '0', 'cpu', '0,1' | Device selection |

### Monitoring Training

```python
# Check training progress
tensorboard --logdir runs/train

# View training plots
# Located in: runs/train/asl_detection/results.png
```

### Training Tips for 96%+ Accuracy

1. **Start with YOLOv8s or YOLOv8m**
   ```python
   train_yolov8_asl(model_size='s', epochs=150)
   ```

2. **Ensure sufficient training time**
   - Minimum 100 epochs
   - Let early stopping decide when to stop

3. **Monitor validation metrics**
   - Watch mAP@0.5 (target: ≥0.96)
   - Check per-class performance

4. **Data quality checks**
   ```python
   from train_yolov8 import verify_dataset
   verify_dataset('data.yaml')
   ```

5. **Handle class imbalance**
   ```python
   from train_yolov8 import train_with_class_weights
   train_with_class_weights(model_size='s', epochs=150)
   ```

---

## 📊 Model Evaluation

### Basic Evaluation

```python
python evaluate_model.py
```

This generates:
- Overall metrics (Precision, Recall, mAP)
- Per-class metrics
- Confusion matrix
- Evaluation report (JSON)

### Evaluate on Test Set

```python
from evaluate_model import evaluate_model

evaluate_model(
    model_path='runs/train/asl_detection/weights/best.pt',
    split='test',  # Use test set instead of validation
    save_dir='runs/eval_test',
)
```

### Compare Multiple Models

```python
from evaluate_model import compare_models

model_paths = [
    'runs/train/exp1/weights/best.pt',
    'runs/train/exp2/weights/best.pt',
    'runs/train/exp3/weights/best.pt',
]

compare_models(model_paths, data_yaml='data.yaml')
```

### Understanding Metrics

**Overall Metrics:**
- **Precision**: Of all predicted signs, what % were correct?
- **Recall**: Of all actual signs, what % were detected?
- **mAP@0.5**: Average precision at 50% IoU threshold (primary metric)
- **mAP@0.5:0.95**: Average precision across IoU 0.5-0.95

**Target:** mAP@0.5 ≥ 0.96 (96%)

**Confusion Matrix:**
- Rows: True labels
- Columns: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

---

## 📦 Model Export

### Export to ONNX (Desktop/Web)

```python
python export_model.py
```

Or programmatically:

```python
from export_model import export_model

export_model(
    model_path='runs/train/asl_detection/weights/best.pt',
    formats=['onnx'],
    imgsz=640,
    simplify=True,
)
```

**Use Cases:**
- Desktop applications (CPU/GPU)
- Web deployment (ONNX.js)
- Cross-platform inference

### Export for Mobile

```python
from export_model import export_for_mobile

export_for_mobile(
    model_path='runs/train/asl_detection/weights/best.pt',
    imgsz=640,
)
```

This creates:
- **TFLite (FP32)**: Best accuracy (~20MB)
- **TFLite (INT8)**: Optimized for mobile (~5MB)
- **CoreML**: For iOS (if on macOS)

**Android Integration:**

```kotlin
// Add to build.gradle
implementation 'org.tensorflow:tensorflow-lite:2.13.0'

// Load model
val tflite = Interpreter(loadModelFile("best.tflite"))

// Inference
val input = preprocessImage(image)
val output = Array(1) { FloatArray(NUM_CLASSES) }
tflite.run(input, output)
```

**iOS Integration:**

```swift
import CoreML

let model = try best_coreml()
let prediction = try model.prediction(image: pixelBuffer)
```

### Export for Edge Devices

```python
from export_model import export_for_edge_devices

export_for_edge_devices(
    model_path='runs/train/asl_detection/weights/best.pt',
    imgsz=640,
)
```

**Raspberry Pi:**
- Use ONNX Runtime
- Reduce imgsz to 416 for better FPS

**NVIDIA Jetson:**
- Use ONNX and convert to TensorRT
- Enable FP16 for 2x speedup

### Benchmark Exported Models

```python
from export_model import benchmark_exported_models

benchmark_exported_models(
    model_path='runs/train/asl_detection/weights/best.pt',
    imgsz=640,
)
```

---

## 🎥 Live Inference

### Basic Webcam Inference

```python
python live_inference.py
```

### Custom Configuration

```python
from live_inference import ASLDetector

detector = ASLDetector(
    model_path='runs/train/asl_detection/weights/best.pt',
    conf_threshold=0.6,           # Confidence threshold
    smoothing_window=15,          # Temporal smoothing window
    min_detection_frames=8,       # Min frames for stable detection
)

detector.run_webcam(
    camera_index=0,               # Camera device
    display_size=(1280, 720),     # Window size
)
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Add detected letter to word |
| `ENTER` | Finalize word and speak sentence |
| `BACKSPACE` | Delete last character |
| `C` | Clear current word |
| `R` | Reset entire sentence |
| `S` | Speak current sentence |
| `Q` | Quit application |

### Phone Camera Setup

**Android (IP Webcam):**

1. Install "IP Webcam" app from Play Store
2. Connect phone and computer to same WiFi
3. Start server in app
4. Note IP address (e.g., 192.168.1.100)

```python
from live_inference import run_from_phone_camera

run_from_phone_camera(
    detector,
    phone_ip='192.168.1.100',
    port=8080,
)
```

**iOS (EpocCam):**

1. Install EpocCam app
2. Install EpocCam drivers on computer
3. Camera will appear as regular webcam

```python
detector.run_webcam(camera_index=1)  # Try different indices
```

### Temporal Smoothing Explained

The system uses temporal filtering to prevent flickering:

1. **Smoothing Window**: Maintains history of last 15 frames
2. **Minimum Frames**: Letter must appear in ≥8 frames to register
3. **Cooldown**: Same letter needs 1.5s wait before re-adding

**Adjust for your needs:**

```python
# More stable but slower response
detector = ASLDetector(
    smoothing_window=20,
    min_detection_frames=12,
)

# Faster response but less stable
detector = ASLDetector(
    smoothing_window=10,
    min_detection_frames=5,
)
```

### TTS Customization

```python
from live_inference import ASLDetector

detector = ASLDetector(model_path='best.pt')

# Adjust TTS settings
detector.tts_engine.setProperty('rate', 150)    # Speed (50-300)
detector.tts_engine.setProperty('volume', 0.9)  # Volume (0-1)

# Change voice
voices = detector.tts_engine.getProperty('voices')
detector.tts_engine.setProperty('voice', voices[1].id)  # Try different voices
```

---

## 🧪 Batch Testing

### Test on Image Directory

```python
from batch_test import BatchTester

tester = BatchTester(
    model_path='runs/train/asl_detection/weights/best.pt',
    conf_threshold=0.5,
)

results = tester.test_images(
    image_dir='test/images',
    save_dir='runs/batch_test',
    visualize=True,
)
```

Output:
- Annotated images in `runs/batch_test/visualizations/`
- JSON report in `runs/batch_test/batch_test_report.json`
- Distribution plot

### Test on Video File

```python
results = tester.test_video(
    video_path='test_video.mp4',
    save_dir='runs/batch_test',
    save_video=True,
)
```

Output:
- Annotated video in `runs/batch_test/annotated_test_video.mp4`
- JSON report with frame-by-frame detections

---

## ⚙️ Advanced Configuration

### Using config.yaml

Edit `config.yaml` to set default parameters:

```yaml
TRAINING:
  MODEL_SIZE: 's'
  EPOCHS: 150
  BATCH_SIZE: 16
  IMAGE_SIZE: 640
  
INFERENCE:
  CONFIDENCE_THRESHOLD: 0.6
  SMOOTHING_WINDOW: 15
  MIN_DETECTION_FRAMES: 8
```

### Custom Dataset

If using a different dataset:

1. Update `data.yaml`:
```yaml
train: path/to/train/images
val: path/to/valid/images
test: path/to/test/images
nc: 26
names: ['A', 'B', ..., 'Z']
```

2. Ensure labels are in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

### Multi-GPU Training

```python
train_yolov8_asl(
    model_size='s',
    device='0,1',  # Use GPU 0 and 1
    batch_size=32,  # Increase batch size
)
```

---

## 🔧 Troubleshooting

### Issue: GPU Not Detected

**Check CUDA:**
```python
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
```

**Reinstall PyTorch:**
```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Low Accuracy (<96%)

1. **Train longer**: Increase epochs to 200+
2. **Use larger model**: Switch to YOLOv8m or YOLOv8l
3. **Check data quality**: Look for labeling errors
4. **Increase resolution**: Use imgsz=800
5. **Analyze confusion matrix**: Identify problem classes

### Issue: Slow Training

1. **Reduce batch size**: Lower to 8 or 4
2. **Use smaller model**: Try YOLOv8n
3. **Check GPU usage**: Should be 90%+
4. **Enable AMP**: Already enabled by default

### Issue: Camera Not Opening

```python
# Test camera
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())

# Try different indices
detector.run_webcam(camera_index=1)
```

### Issue: TTS Not Working

```powershell
# Test TTS
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('test'); engine.runAndWait()"

# Alternative: Use gTTS
pip install gTTS
```

---

## ⚡ Performance Tips

### Training Performance

1. **Use appropriate batch size:**
   - RTX 3080 (10GB): batch_size=16-32
   - RTX 3060 (12GB): batch_size=16
   - GTX 1660 (6GB): batch_size=8

2. **Enable caching:**
   ```python
   train_yolov8_asl(cache='ram')  # Cache in RAM (faster)
   ```

3. **Use mixed precision:**
   - Already enabled by default (amp=True)

### Inference Performance

1. **Use ONNX for faster inference:**
   ```python
   export_model(formats=['onnx'])
   ```

2. **Reduce image size:**
   ```python
   detector = ASLDetector(imgsz=416)  # Smaller, faster
   ```

3. **Lower confidence threshold:**
   ```python
   detector = ASLDetector(conf_threshold=0.5)
   ```

### Expected Performance

**Training Time (RTX 3080):**
- YOLOv8n: ~30 min (150 epochs)
- YOLOv8s: ~45 min (150 epochs)
- YOLOv8m: ~90 min (150 epochs)

**Inference Speed:**
- YOLOv8n: 100+ FPS (GPU), 25+ FPS (CPU)
- YOLOv8s: 80+ FPS (GPU), 15+ FPS (CPU)
- YOLOv8m: 60+ FPS (GPU), 8+ FPS (CPU)

---

## 📝 Best Practices

### Data Collection

1. Use consistent lighting
2. Plain backgrounds
3. Multiple angles per sign
4. Various hand sizes and skin tones
5. Different environments

### Labeling

1. Use proper bounding boxes
2. Label consistently
3. Double-check labels
4. Balance classes

### Training

1. Start with small model (fast iteration)
2. Monitor validation metrics
3. Use early stopping
4. Save best model

### Deployment

1. Export to appropriate format
2. Test on target device
3. Optimize for speed vs accuracy
4. Monitor real-world performance

---

## 🎯 Project Checklist

- [ ] Install dependencies
- [ ] Verify dataset structure
- [ ] Train model (target: mAP@0.5 ≥ 0.96)
- [ ] Evaluate on test set
- [ ] Export to ONNX/TFLite
- [ ] Test live inference
- [ ] Deploy to target platform
- [ ] Document custom configurations

---

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)

---

**Happy Coding! 🤟**
