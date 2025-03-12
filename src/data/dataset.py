import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path


class WeaklySegDataset(Dataset):
    # List of sample images to debug (these should match the ones in train.py)
    SAMPLE_INDICES = [0, 1, 2]  # First three images for debugging

    def __init__(self, root_dir, split='train', debug_mode=False):
        """
        Dataset class for weakly supervised nuclei segmentation.
        Only provides point annotations for each nuclei.

        Args:
            root_dir (str): Path to the dataset directory
            split (str): 'train' or 'val'
            debug_mode (bool): If True, only loads and processes sample images
        """
        self.root_dir = root_dir
        self.split = split
        self.debug_mode = debug_mode
        self.image_size = 256
        self.point_radius = 3  # radius of point annotation

        # Setup transforms
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Setup mask transforms (without normalization)
        self.mask_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            ToTensorV2()
        ])

        # Load image paths and verify matching XML files
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')

        # Get all image files
        image_files = [f for f in sorted(os.listdir(self.image_dir)) if f.endswith('.tif')]

        # Verify matching XML files exist
        self.image_mask_pairs = []
        print(f"\nVerifying image-mask pairs for {split} set:")
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]  # Remove .tif extension
            xml_file = f"{base_name}.xml"
            xml_path = os.path.join(self.mask_dir, xml_file)

            if os.path.exists(xml_path):
                self.image_mask_pairs.append((img_file, xml_file))
                print(f"✓ Found pair: {img_file} -> {xml_file}")
            else:
                print(f"✗ Missing XML for: {img_file}")

        # If in debug mode, only keep the sample images
        if debug_mode:
            self.image_mask_pairs = [self.image_mask_pairs[i] for i in self.SAMPLE_INDICES if i < len(self.image_mask_pairs)]
            print(f"\nDebug mode: Using {len(self.image_mask_pairs)} sample images")
        else:
            print(f"\nTotal valid pairs found: {len(self.image_mask_pairs)}/{len(image_files)}\n")

        # Create debug directory
        self.debug_dir = os.path.join(root_dir, 'debug_masks')
        os.makedirs(self.debug_dir, exist_ok=True)

        self.precomputed_masks = []
        for img_file, xml_file in self.image_mask_pairs:
            img_path = os.path.join(self.image_dir, img_file)
            xml_path = os.path.join(self.mask_dir, xml_file)

            # Load image to get its shape
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Precompute the mask
            full_mask, point_mask = self._xml_to_mask(xml_path, image.shape, img_file)
            self.precomputed_masks.append((full_mask, point_mask))

    def _create_point_mask(self, full_mask):
        """Create point annotations from full mask by finding centroids of each nucleus"""
        point_mask = np.zeros_like(full_mask)

        # Label connected components in the mask
        labeled_mask, num_labels = ndimage.label(full_mask)

        # Find centroid of each nucleus
        for label in range(1, num_labels + 1):
            nucleus = labeled_mask == label
            centroid = ndimage.center_of_mass(nucleus)
            y, x = int(centroid[0]), int(centroid[1])

            # Draw a small circle at the centroid
            cv2.circle(point_mask, (x, y), self.point_radius, 1, -1)

        return point_mask

    def _xml_to_mask(self, xml_path, image_shape, img_name=None):
        """Convert XML annotations to binary mask."""
        mask = np.zeros(image_shape[:2], dtype=np.float32)

        if not os.path.exists(xml_path):
            print(f"Warning: XML file not found: {xml_path}")
            return mask, mask

        tree = ET.parse(xml_path)
        root = tree.getroot()

        try:
            # Get image dimensions from XML
            regions = root.findall('.//Region')
            #print(f"Processing {img_name}: Found {len(regions)} regions")

            for region in regions:
                vertices = region.findall('.//Vertex')
                if len(vertices) > 0:
                    # Extract coordinates
                    coords = []
                    for vertex in vertices:
                        # XML coordinates are in the original image space
                        x = float(vertex.get('X'))
                        y = float(vertex.get('Y'))

                        # Scale coordinates to match image dimensions
                        x = int(x * mask.shape[1] / image_shape[1])
                        y = int(y * mask.shape[0] / image_shape[0])

                        coords.append([x, y])

                    coords = np.array(coords, dtype=np.int32)

                    # Fill polygon
                    cv2.fillPoly(mask, [coords], 1)

            # Create point annotations from the full mask
            point_mask = self._create_point_mask(mask)

            folder_path = Path(self.debug_dir)  # Kendi klasör yolunla değiştir
            file_count = len(list(folder_path.glob('*')))  # Sadece dosyaları sayar

            # Debug visualization
            if file_count < 9 and img_name is not None and (self.debug_mode or any(img_name == self.image_mask_pairs[i][0] for i in self.SAMPLE_INDICES)):
                plt.figure(figsize=(10, 10))

                # Plot original image (top-left)
                plt.subplot(221)
                img_path = os.path.join(self.image_dir, img_name)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.title('Original Image')
                plt.axis('off')

                # Plot point annotations (top-right)
                plt.subplot(222)
                plt.imshow(img)
                plt.imshow(point_mask, alpha=0.7, cmap='Reds')
                plt.title('Weak Supervision (Points)')
                plt.axis('off')

                # Plot full mask (bottom-left)
                plt.subplot(223)
                plt.imshow(img)
                plt.imshow(mask, alpha=0.5, cmap='Reds')
                plt.title('Ground Truth (Full)')
                plt.axis('off')

                # Plot prediction placeholder (bottom-right)
                plt.subplot(224)
                plt.imshow(np.zeros_like(img))
                plt.title('Prediction (To be filled)')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(self.debug_dir, f'debug_{img_name}.png'))
                plt.close()

                # Save masks separately for detailed inspection
                cv2.imwrite(os.path.join(self.debug_dir, f'mask_full_{img_name}.png'), (mask * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(self.debug_dir, f'mask_points_{img_name}.png'), (point_mask * 255).astype(np.uint8))

        except Exception as e:
            print(f"Error processing XML file {xml_path}: {str(e)}")
            return np.zeros(image_shape[:2], dtype=np.float32), np.zeros(image_shape[:2], dtype=np.float32)

        return mask, point_mask

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        # Load image
        img_file, _ = self.image_mask_pairs[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get precomputed masks
        full_mask, point_mask = self.precomputed_masks[idx]

        # Apply transformations
        transformed = self.transform(image=image, mask=point_mask)
        image = transformed['image']
        point_mask = transformed['mask']

        return image, point_mask

def get_data_loaders(root_dir, batch_size=8, debug_mode=False):
    """Create data loaders for training and validation"""
    train_dataset = WeaklySegDataset(root_dir, split='train', debug_mode=debug_mode)
    val_dataset = WeaklySegDataset(root_dir, split='val', debug_mode=debug_mode)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=not debug_mode, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader