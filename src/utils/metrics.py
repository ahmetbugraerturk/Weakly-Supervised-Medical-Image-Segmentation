import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

def calculate_metrics(pred, target):
    """Calculate metrics for binary segmentation."""
    # Convert to numpy arrays
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Flatten and binarize predictions and targets
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    
    # Convert to binary
    pred_binary = (pred > 0.5).astype(np.int32)
    target_binary = (target > 0.5).astype(np.int32)
    
    return {
        'accuracy': accuracy_score(target_binary, pred_binary),
        'precision': precision_score(target_binary, pred_binary, zero_division=0),
        'recall': recall_score(target_binary, pred_binary, zero_division=0),
        'f1': f1_score(target_binary, pred_binary, zero_division=0)
    }

def save_prediction_image(image, mask, pred, save_path):
    """Save a figure showing the original image, ground truth mask, and prediction."""
    # Convert image from tensor to numpy and denormalize
    image = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    # Convert mask and prediction to numpy
    mask = mask.cpu().numpy().squeeze()
    pred = pred.cpu().numpy().squeeze()
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot ground truth mask
    ax2.imshow(image)
    ax2.imshow(mask, cmap='gray', vmin=0, vmax=1, alpha=0.7)
    ax2.set_title('Ground Truth Mask')
    ax2.axis('off')
    
    # Plot prediction
    ax3.imshow(image)
    ax3.imshow(pred, cmap='gray', vmin=0, vmax=1, alpha=0.5)
    ax3.set_title(f'Prediction (IoU: {calculate_iou(mask, pred):.2%})')
    ax3.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_iou(mask, pred):
    """Calculate Intersection over Union."""
    mask = mask > 0.5
    pred = pred > 0.5
    intersection = np.logical_and(mask, pred).sum()
    union = np.logical_or(mask, pred).sum()
    return intersection / (union + 1e-6) 