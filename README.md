# PlateSight - Malaysian Food Recognition System

An intelligent web application that leverages computer vision and deep learning to automatically identify Malaysian cuisine. Built with modern web technologies for scalable deployment and cross-platform accessibility.

## 🎯 Overview

PlateSight is a production-ready food recognition system that accurately identifies 17 traditional Malaysian dishes using state-of-the-art YOLO (You Only Look Once) object detection models. The application provides real-time analysis through an intuitive web interface.

**Key Capabilities:**
- Multi-modal input support (images, videos, live camera)
- Real-time object detection with confidence scoring
- Responsive web interface built with Streamlit
- Multiple pre-trained models for optimal performance
- Scalable architecture suitable for production deployment

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **ML Models**: YOLO v11/v12, Custom Roboflow model
- **Computer Vision**: OpenCV, Ultralytics
- **Backend**: PyTorch, NumPy
- **Deployment**: Docker-ready, cloud-compatible

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser

### Installation & Setup

```bash
# Clone repository
git clone https://github.com/[username]/PlateSight.git
cd PlateSight

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run main.py
```

Access the application at `http://localhost:8501`

## 📋 Supported Food Categories

The system recognizes 17 traditional Malaysian food items:

**Main Dishes**: Char-Kuey-Teow, Chicken-Rendang, Fried-Rice, Hokkien-Mee, Lo-Mein, Mee-Rebus, Mee-Siam, Roti-Canai

**Components**: Rice, Fried-Chicken, Fried-Egg, Boiled-Egg, Curry-Puff

**Accompaniments**: Anchovies, Peanuts, Sambal, Slices-Cucumber

## 💼 Usage Instructions

1. **Model Selection**: Choose from available YOLO models in the sidebar
2. **Input Method**: Upload images/videos or use live camera feed
3. **Parameter Tuning**: Adjust confidence threshold for optimal results
4. **Analysis**: View real-time detection results with bounding boxes and confidence scores

## ⚡ Performance Optimization

- **Model Caching**: Intelligent caching system reduces load times
- **Frame Processing**: Configurable frame skipping for video analysis
- **Real-time Processing**: Optimized inference pipeline for live feeds
- **Responsive Design**: Adaptive interface for various screen sizes

## 🔧 Configuration

The application supports multiple detection models:
- **YOLO v11**: Balanced speed and accuracy
- **YOLO v12**: Latest architecture with enhanced performance  
- **Roboflow 3.0**: Domain-specific model trained on Malaysian cuisine

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Application fails to start | Verify Python 3.8+ installation and dependencies |
| No detections found | Adjust confidence threshold or improve image lighting |
| Camera access denied | Enable camera permissions in browser settings |

## 📄 License & Attribution

This project utilizes open-source models and frameworks. Model weights are trained on datasets licensed under Creative Commons.

**Made with ❤️ for Malaysian food lovers**
