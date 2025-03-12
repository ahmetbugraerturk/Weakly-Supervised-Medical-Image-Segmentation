import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import UNet
from data.dataset import get_data_loaders
from utils.metrics import calculate_metrics, save_prediction_image

def dice_loss(pred, target):
    smooth = 1.0
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def plot_losses(train_losses, val_losses, output_dir):
    plt.figure()
    plt.plot(train_losses, 'r-', label='Train Loss')
    plt.plot(val_losses, 'orange', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

def train(data_dir, output_dir, num_epochs=100, batch_size=8, learning_rate=1e-4):
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = UNet().to(device)
    
    # Setup data loaders
    train_loader, val_loader = get_data_loaders(data_dir, batch_size)
    
    # Loss function and optimizer
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate metrics
            batch_metrics = calculate_metrics(outputs.detach(), masks.detach())
            for k, v in batch_metrics.items():
                train_metrics[k] += v
        
        # Average training metrics
        train_loss /= len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
            writer.add_scalar(f'Train/{k}', train_metrics[k], epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                batch_metrics = calculate_metrics(outputs, masks)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v
        
        # Average validation metrics
        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
            writer.add_scalar(f'Val/{k}', val_metrics[k], epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        val_losses.append(val_loss)

        # Print progress
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, F1: {train_metrics["f1"]:.4f}')
        print(f'Val Loss: {val_loss:.4f}, F1: {val_metrics["f1"]:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
        
        # Save sample predictions every epoch
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                images, masks = next(iter(val_loader))
                images = images.to(device)
                outputs = model(images)
                
                for i in range(min(3, len(images))):
                    save_prediction_image(
                        images[i],
                        masks[i],
                        outputs[i],
                        os.path.join(output_dir, 'predictions', f'epoch_{epoch+1}_sample_{i}.png')
                    )
    plot_losses(train_losses, val_losses, output_dir)
if __name__ == '__main__':
    data_dir = 'data'
    output_dir = 'outputs/nuclei_segmentation'
    train(data_dir, output_dir) 