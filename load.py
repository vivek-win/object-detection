import torch
import torchvision.transforms as transforms
from torchvision.models import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64

class ObjectDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # COCO class names
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
            'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
            'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def detect_objects(self, image_path, confidence_threshold=0.5):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process results
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        keep = scores >= confidence_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        # Create result image with bounding boxes
        result_image_path = self._draw_bounding_boxes(image, boxes, labels, scores, image_path)
        
        # Prepare results
        results = []
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            results.append({
                'class': self.class_names[label],
                'confidence': float(score),
                'bbox': box.tolist()
            })
        
        return {
            'detections': results,
            'result_image': result_image_path.split('/')[-1]  # Return only filename
        }
    
    def _draw_bounding_boxes(self, image, boxes, labels, scores, original_path):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            color = colors[i % len(colors)]
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            label_text = f'{self.class_names[label]}: {score:.2f}'
            ax.text(x1, y1-10, label_text, fontsize=10, color=color, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save result image
        result_filename = f"result_{original_path.split('/')[-1]}"
        result_path = f"static/uploads/{result_filename}"
        plt.savefig(result_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        return result_path
