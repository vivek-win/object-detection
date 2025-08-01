# object-detection
# Object Detection from Uploaded Images

A web-based object detection tool built with Flask and Faster R-CNN from Torchvision that can detect objects in uploaded images.

## Features

- 🖼️ **Image Upload**: Upload images through a clean web interface
- 🔍 **Object Detection**: Uses pre-trained Faster R-CNN model for accurate object detection
- 📦 **Bounding Boxes**: Visual results with bounding boxes around detected objects
- 🎨 **Matplotlib Visualization**: Clean visual overlays using Matplotlib
- 🌐 **Minimal UI**: Simple HTML/CSS interface for easy use

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Deep Learning**: PyTorch, Torchvision (Faster R-CNN)
- **Image Processing**: Pillow (PIL), NumPy
- **Visualization**: Matplotlib
- **Frontend**: HTML, CSS, JavaScript

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection-webapp.git
cd object-detection-webapp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
