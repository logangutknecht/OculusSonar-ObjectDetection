"""
Training Script for ML-based Sonar Object Detection
Uses YOLOv8 with transfer learning optimized for sonar imagery
"""

import os
import sys
from pathlib import Path
import yaml
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import torch
from ultralytics import YOLO
import logging
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SonarDataPreprocessor:
    """Preprocesses sonar images for optimal ML training"""
    
    @staticmethod
    def enhance_sonar_image(image_path: str, output_path: str) -> bool:
        """
        Apply sonar-specific preprocessing to improve ML training
        
        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image
            
        Returns:
            True if successful
        """
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This is particularly effective for sonar images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)
            
            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Normalize to 0-255 range
            normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert to 3-channel for YOLO (RGB format)
            # For sonar, we replicate the grayscale channel
            rgb_image = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
            
            # Save enhanced image
            cv2.imwrite(output_path, rgb_image)
            return True
            
        except Exception as e:
            logger.error(f"Error enhancing image {image_path}: {e}")
            return False
    
    @staticmethod
    def augment_sonar_data(image: np.ndarray, boxes: List[List[float]], 
                           augmentation_type: str = "random") -> Tuple[np.ndarray, List[List[float]]]:
        """
        Apply sonar-specific data augmentation
        
        Args:
            image: Input image
            boxes: Bounding boxes in YOLO format [class_id, cx, cy, w, h]
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented image and boxes
        """
        h, w = image.shape[:2]
        augmented_img = image.copy()
        augmented_boxes = [box.copy() for box in boxes]
        
        if augmentation_type == "random":
            augmentation_type = np.random.choice(["noise", "intensity", "blur", "flip"])
        
        if augmentation_type == "noise":
            # Add Gaussian noise (common in sonar)
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            augmented_img = cv2.add(augmented_img, noise)
            
        elif augmentation_type == "intensity":
            # Random intensity adjustment (simulates different sonar gain settings)
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.uniform(-20, 20)
            augmented_img = cv2.convertScaleAbs(augmented_img, alpha=alpha, beta=beta)
            
        elif augmentation_type == "blur":
            # Motion blur (simulates platform movement)
            kernel_size = np.random.randint(3, 7)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            augmented_img = cv2.filter2D(augmented_img, -1, kernel)
            
        elif augmentation_type == "flip":
            # Horizontal flip (valid for sonar data)
            augmented_img = cv2.flip(augmented_img, 1)
            # Update box coordinates
            for box in augmented_boxes:
                box[1] = 1.0 - box[1]  # Flip x-coordinate
        
        return augmented_img, augmented_boxes


class SonarModelTrainer:
    """Handles training of YOLO model for sonar object detection"""
    
    def __init__(self, data_dir: str, project_name: str = "sonar_detection"):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing annotated data
            project_name: Name for the training project
        """
        self.data_dir = Path(data_dir)
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("training_runs") / f"{project_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters optimized for sonar data
        self.config = {
            'model_size': 'yolov8n.pt',  # Nano model - good for starting
            'epochs': 100,  # Will use early stopping
            'batch_size': 16,
            'imgsz': 640,  # Image size
            'patience': 20,  # Early stopping patience
            'conf_threshold': 0.25,  # Lower threshold for sonar (harder to detect)
            'iou_threshold': 0.45,
            'learning_rate': 0.01,
            'warmup_epochs': 3,
            'mosaic': 0.5,  # Mosaic augmentation probability
            'mixup': 0.1,   # Mixup augmentation probability
            'copy_paste': 0.0,  # Copy-paste augmentation (disabled for sonar)
            'degrees': 0,  # No rotation for sonar (orientation matters)
            'translate': 0.1,  # Small translation
            'scale': 0.2,  # Small scale variation
            'shear': 0,  # No shear for sonar
            'perspective': 0.0,  # No perspective change
            'flipud': 0.0,  # No vertical flip for sonar
            'fliplr': 0.5,  # Horizontal flip okay
            'hsv_h': 0,  # No hue shift (grayscale)
            'hsv_s': 0,  # No saturation shift (grayscale)
            'hsv_v': 0.3,  # Value (brightness) variation
        }
        
        self.model = None
        self.results = None
        
    def prepare_dataset(self, train_split: float = 0.8, preprocess: bool = True):
        """
        Prepare dataset for training
        
        Args:
            train_split: Fraction of data for training (rest for validation)
            preprocess: Whether to apply sonar-specific preprocessing
        """
        logger.info("Preparing dataset...")
        
        # Check for required directories
        images_dir = self.data_dir / 'images'
        labels_dir = self.data_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError(f"Images or labels directory not found in {self.data_dir}")
        
        # Get all images
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
        
        if not image_files:
            raise ValueError(f"No images found in {images_dir}")
        
        logger.info(f"Found {len(image_files)} images")
        
        # Split into train and validation
        train_images, val_images = train_test_split(
            image_files, train_size=train_split, random_state=42
        )
        
        # Create YOLO dataset structure
        dataset_dir = self.output_dir / 'dataset'
        train_img_dir = dataset_dir / 'train' / 'images'
        train_label_dir = dataset_dir / 'train' / 'labels'
        val_img_dir = dataset_dir / 'val' / 'images'
        val_label_dir = dataset_dir / 'val' / 'labels'
        
        for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Process and copy files
        preprocessor = SonarDataPreprocessor()
        
        def process_split(image_list, img_dir, label_dir, split_name):
            logger.info(f"Processing {split_name} split ({len(image_list)} images)...")
            
            for img_path in image_list:
                # Process image
                output_img_path = img_dir / img_path.name
                
                if preprocess:
                    preprocessor.enhance_sonar_image(str(img_path), str(output_img_path))
                else:
                    shutil.copy2(img_path, output_img_path)
                
                # Copy corresponding label
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, label_dir / label_path.name)
        
        process_split(train_images, train_img_dir, train_label_dir, "training")
        process_split(val_images, val_img_dir, val_label_dir, "validation")
        
        # Create data.yaml for YOLO
        classes_file = self.data_dir / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            class_names = ['object']  # Default
        
        data_yaml = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)
        }
        
        yaml_path = dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        logger.info(f"Dataset prepared: {len(train_images)} train, {len(val_images)} val")
        logger.info(f"Classes: {class_names}")
        
        return yaml_path
    
    def train(self, data_yaml_path: Optional[str] = None, resume: bool = False):
        """
        Train the model
        
        Args:
            data_yaml_path: Path to data configuration file
            resume: Whether to resume from previous training
        """
        if data_yaml_path is None:
            data_yaml_path = self.prepare_dataset()
        
        logger.info("Starting training...")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize model
        if resume and (self.output_dir / 'weights' / 'last.pt').exists():
            model_path = self.output_dir / 'weights' / 'last.pt'
            logger.info(f"Resuming from {model_path}")
        else:
            model_path = self.config['model_size']
            logger.info(f"Starting with pretrained model: {model_path}")
        
        self.model = YOLO(model_path)
        
        # Train
        self.results = self.model.train(
            data=data_yaml_path,
            epochs=self.config['epochs'],
            imgsz=self.config['imgsz'],
            batch=self.config['batch_size'],
            patience=self.config['patience'],
            save=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            project=str(self.output_dir),
            name='train',
            exist_ok=True,
            pretrained=True,
            optimizer='SGD',
            lr0=self.config['learning_rate'],
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=self.config['warmup_epochs'],
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,  # Box loss gain
            cls=0.5,  # Class loss gain  
            dfl=1.5,  # DFL loss gain
            conf=self.config['conf_threshold'],
            iou=self.config['iou_threshold'],
            # Augmentation parameters
            hsv_h=self.config['hsv_h'],
            hsv_s=self.config['hsv_s'], 
            hsv_v=self.config['hsv_v'],
            degrees=self.config['degrees'],
            translate=self.config['translate'],
            scale=self.config['scale'],
            shear=self.config['shear'],
            perspective=self.config['perspective'],
            flipud=self.config['flipud'],
            fliplr=self.config['fliplr'],
            mosaic=self.config['mosaic'],
            mixup=self.config['mixup'],
            copy_paste=self.config['copy_paste'],
            auto_augment='randaugment',  # Use RandAugment
            erasing=0.1,  # Random erasing probability
            crop_fraction=1.0,  # No cropping
        )
        
        logger.info("Training completed!")
        
        # Save configuration
        config_path = self.output_dir / 'training_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        return self.results
    
    def evaluate(self, test_data_path: Optional[str] = None):
        """
        Evaluate the trained model
        
        Args:
            test_data_path: Path to test dataset (uses validation set if None)
        """
        if self.model is None:
            # Load best model
            best_model_path = self.output_dir / 'train' / 'weights' / 'best.pt'
            if not best_model_path.exists():
                logger.error("No trained model found")
                return None
            
            self.model = YOLO(best_model_path)
        
        logger.info("Evaluating model...")
        
        # Use validation set if no test set provided
        if test_data_path is None:
            dataset_dir = self.output_dir / 'dataset'
            data_yaml_path = dataset_dir / 'data.yaml'
            
            if not data_yaml_path.exists():
                logger.error("No dataset found for evaluation")
                return None
            
            metrics = self.model.val(data=str(data_yaml_path))
        else:
            metrics = self.model.val(data=test_data_path)
        
        # Print metrics
        logger.info("\nEvaluation Results:")
        logger.info(f"  mAP50: {metrics.box.map50:.3f}")
        logger.info(f"  mAP50-95: {metrics.box.map:.3f}")
        logger.info(f"  Precision: {metrics.box.mp:.3f}")
        logger.info(f"  Recall: {metrics.box.mr:.3f}")
        
        # Save metrics
        metrics_dict = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }
        
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        return metrics
    
    def export_model(self, format: str = 'onnx'):
        """
        Export trained model to different formats
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
        """
        if self.model is None:
            best_model_path = self.output_dir / 'train' / 'weights' / 'best.pt'
            if not best_model_path.exists():
                logger.error("No trained model found")
                return None
            
            self.model = YOLO(best_model_path)
        
        logger.info(f"Exporting model to {format} format...")
        
        # Export model
        export_path = self.model.export(format=format)
        
        logger.info(f"Model exported to: {export_path}")
        
        # Copy to models directory for easy access
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        export_name = f"sonar_detector_{self.timestamp}.{format}"
        final_path = models_dir / export_name
        
        shutil.copy2(export_path, final_path)
        logger.info(f"Model copied to: {final_path}")
        
        return final_path


def train_from_scratch(data_directory: str, **kwargs):
    """
    Convenience function to train a model from scratch
    
    Args:
        data_directory: Directory containing annotated data
        **kwargs: Additional training parameters
    """
    trainer = SonarModelTrainer(data_directory)
    
    # Update config with any provided parameters
    trainer.config.update(kwargs)
    
    # Train model
    trainer.train()
    
    # Evaluate
    trainer.evaluate()
    
    # Export to ONNX for deployment
    trainer.export_model('onnx')
    
    # Export to PyTorch format for integration
    best_model = trainer.output_dir / 'train' / 'weights' / 'best.pt'
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    final_model = models_dir / f"sonar_yolo_best_{trainer.timestamp}.pt"
    shutil.copy2(best_model, final_model)
    
    logger.info(f"\nTraining complete! Best model saved to: {final_model}")
    logger.info(f"Full results in: {trainer.output_dir}")
    
    return str(final_model)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 for sonar object detection")
    parser.add_argument('--data', type=str, default='training_data',
                       help='Path to annotated data directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous training')
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_from_scratch(
        args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.img_size
    )
    
    print(f"\nModel ready for use: {model_path}")
    print("To use in your pipeline, update config.yaml with:")
    print(f"  ml_model_path: {model_path}")
