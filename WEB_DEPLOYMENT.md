# ASL to Voice - Web Deployment Guide

## 🌐 Local Testing

### 1. Install Flask
```powershell
pip install flask flask-cors pillow
```

### 2. Run the Web Server
```powershell
python app.py
```

### 3. Open Browser
Navigate to: `http://localhost:5000`

---

## 🚀 Public Deployment Options

### Option 1: Deploy on Heroku (Free Tier Available)

1. **Install Heroku CLI**
   ```powershell
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Procfile**
   ```
   web: gunicorn app:app
   ```

3. **Create requirements.txt**
   ```powershell
   pip freeze > requirements.txt
   ```

4. **Deploy**
   ```powershell
   heroku login
   heroku create asl-to-voice-app
   git init
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

### Option 2: Deploy on Render (Recommended - Free with GPU support)

1. **Create `render.yaml`**
   ```yaml
   services:
     - type: web
       name: asl-to-voice
       env: python
       buildCommand: "pip install -r requirements.txt"
       startCommand: "gunicorn app:app"
   ```

2. **Push to GitHub**
   ```powershell
   git init
   git add .
   git commit -m "ASL to Voice web app"
   git push origin main
   ```

3. **Connect Render to GitHub**
   - Go to https://render.com
   - Create new Web Service
   - Connect your repository
   - Deploy automatically

### Option 3: Deploy on Google Cloud Run

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
   ```

2. **Deploy**
   ```powershell
   gcloud run deploy asl-to-voice --source .
   ```

### Option 4: Deploy on AWS EC2

1. **Launch EC2 instance** (t2.medium or higher)
2. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip nginx
   pip3 install -r requirements.txt
   ```

3. **Configure Nginx** as reverse proxy
4. **Use systemd** to run Flask app

---

## 📦 Additional Dependencies for Production

```powershell
pip install gunicorn eventlet
```

Update `app.py` last line:
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
```

---

## 🔒 Security Considerations

1. **Rate Limiting**: Add Flask-Limiter
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, default_limits=["200 per day", "50 per hour"])
   ```

2. **HTTPS**: Use Let's Encrypt or cloud provider SSL

3. **Environment Variables**: Store model paths in `.env`

4. **CORS**: Restrict origins in production
   ```python
   CORS(app, origins=["https://yourdomain.com"])
   ```

---

## 🎯 Features

✅ Real-time ASL detection via webcam
✅ Temporal smoothing for stable predictions
✅ Word and sentence assembly
✅ Browser-based text-to-speech
✅ Responsive UI for mobile/desktop
✅ Keyboard shortcuts
✅ Punctuation support (period, comma, exclamation)

---

## 🛠️ Troubleshooting

**Issue**: Model loading slow
**Solution**: Use model quantization or ONNX export

**Issue**: High latency
**Solution**: Reduce frame processing rate or use smaller model

**Issue**: Webcam not working
**Solution**: Ensure HTTPS (required for webcam in most browsers)

---

## 📊 Performance Tips

1. **Use ONNX model** for faster inference
2. **Implement caching** for repeated detections
3. **Use WebSockets** instead of polling for real-time updates
4. **Optimize image size** before sending to backend
5. **Enable GPU** on cloud instances

---

## 🌟 Next Steps

1. Add user authentication
2. Save user sessions/history
3. Add sign language learning mode
4. Support multiple languages
5. Add video recording feature
6. Create mobile app version
