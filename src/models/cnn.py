import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = DoubleConv(3, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dconv4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv1 = DoubleConv(128, 64)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        x = self.pool(conv1)
        
        conv2 = self.conv2(x)
        x = self.pool(conv2)
        
        conv3 = self.conv3(x)
        x = self.pool(conv3)
        
        conv4 = self.conv4(x)
        x = self.pool(conv4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv1(x)
        
        # Final convolution
        x = torch.sigmoid(self.final_conv(x))
        
        return x 