"""
Quick Training Script for Sonar Object Detection
Automatically trains YOLOv8 on your annotated data
"""

import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append('src')

def check_dependencies():
    """Check if required packages are installed"""
    missing_packages = []
    
    try:
        import ultralytics
        logger.info(f"✓ ultralytics installed (version {ultralytics.__version__})")
    except ImportError:
        missing_packages.append('ultralytics')
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "CUDA" if cuda_available else "CPU"
        logger.info(f"✓ PyTorch installed (using {device})")
    except ImportError:
        missing_packages.append('torch')
    
    try:
        import cv2
        logger.info("✓ OpenCV installed")
    except ImportError:
        missing_packages.append('opencv-python')
    
    try:
        import yaml
        logger.info("✓ PyYAML installed")
    except ImportError:
        missing_packages.append('pyyaml')
    
    try:
        import matplotlib
        logger.info("✓ Matplotlib installed")
    except ImportError:
        missing_packages.append('matplotlib')
    
    try:
        import sklearn
        logger.info("✓ scikit-learn installed")
    except ImportError:
        missing_packages.append('scikit-learn')
    
    if missing_packages:
        print("\n❌ Missing required packages:")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def train_quick(data_dir='training_data/annotations', 
                epochs=50,
                batch_size=8,
                model_size='n'):
    """
    Quick training with sensible defaults
    
    Args:
        data_dir: Directory with annotated data
        epochs: Number of training epochs (default 50 for quick training)
        batch_size: Batch size (default 8 for memory efficiency)
        model_size: YOLO model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large)
    """
    from ml_detector.train_sonar_detector import SonarModelTrainer
    
    print("\n" + "="*60)
    print("🚀 SONAR OBJECT DETECTION TRAINING")
    print("="*60)
    
    # Check data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Count images
    images_dir = data_path / 'images'
    if images_dir.exists():
        image_count = len(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Total images: {image_count}")
        print(f"   • Training images: ~{int(image_count * 0.8)}")
        print(f"   • Validation images: ~{int(image_count * 0.2)}")
    
    # Check classes
    classes_file = data_path / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"   • Classes: {', '.join(classes)}")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   • Model size: YOLOv8{model_size}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    
    # Ask for confirmation
    response = input("\n▶️  Start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Initialize trainer
    print("\n🔧 Initializing trainer...")
    trainer = SonarModelTrainer(data_dir)
    
    # Update configuration for sonar-specific training
    trainer.config.update({
        'model_size': f'yolov8{model_size}.pt',
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': max(10, epochs // 5),  # Early stopping patience
        'conf_threshold': 0.25,  # Lower threshold for sonar
        'iou_threshold': 0.45,
        'learning_rate': 0.01,
        'imgsz': 640,  # Standard YOLO input size
    })
    
    # Prepare dataset
    print("\n📁 Preparing dataset...")
    data_yaml_path = trainer.prepare_dataset(
        train_split=0.8,  # 80% train, 20% validation
        preprocess=True    # Apply sonar-specific preprocessing
    )
    
    # Train model
    print("\n🏋️ Starting training...")
    print("This may take a while depending on your hardware...")
    print("You can monitor progress in the output below.\n")
    
    try:
        results = trainer.train(data_yaml_path)
        
        # Evaluate model
        print("\n📈 Evaluating model...")
        metrics = trainer.evaluate()
        
        # Export model
        print("\n💾 Exporting model...")
        
        # Export to PyTorch format for use in pipeline
        best_model = trainer.output_dir / 'train' / 'weights' / 'best.pt'
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        final_model = models_dir / f"sonar_yolo_best_{trainer.timestamp}.pt"
        import shutil
        shutil.copy2(best_model, final_model)
        
        # Also export ONNX for deployment
        onnx_path = trainer.export_model('onnx')
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"\n📊 Final Metrics:")
        print(f"   • mAP@50: {metrics.box.map50:.3f}")
        print(f"   • mAP@50-95: {metrics.box.map:.3f}")
        print(f"   • Precision: {metrics.box.mp:.3f}")
        print(f"   • Recall: {metrics.box.mr:.3f}")
        
        print(f"\n📁 Output Files:")
        print(f"   • Best model: {final_model}")
        print(f"   • ONNX model: {onnx_path}")
        print(f"   • Full results: {trainer.output_dir}")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. To use the model in your pipeline, update config.yaml:")
        print(f"   detection:")
        print(f"     use_ml: true")
        print(f"     ml_model_path: {final_model}")
        print(f"")
        print(f"2. Test the model on a sonar file:")
        print(f"   python main.py raw_assets/your_file.oculus --config config.yaml")
        print(f"")
        print(f"3. For better results, annotate more images and retrain!")
        
        return str(final_model)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print("\n❌ Training failed. Common issues:")
        print("  • Out of memory: Try reducing batch_size")
        print("  • CUDA error: Try using CPU by setting device='cpu'")
        print("  • Missing annotations: Check that all images have label files")
        return None

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model on sonar data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with defaults (recommended for first run)
  python train_model.py
  
  # Custom epochs and batch size
  python train_model.py --epochs 100 --batch-size 16
  
  # Use larger model for better accuracy
  python train_model.py --model-size m
  
  # Use custom data directory
  python train_model.py --data training_data/my_annotations
        """
    )
    
    parser.add_argument('--data', type=str, 
                       default='training_data/annotations',
                       help='Path to annotations directory (default: training_data/annotations)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8, reduce if out of memory)')
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n=nano(fast), s=small, m=medium, l=large, x=xlarge(best)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check dependencies without training')
    
    args = parser.parse_args()
    
    # Check dependencies first
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing packages before training.")
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ All dependencies are installed!")
        sys.exit(0)
    
    # Start training
    model_path = train_quick(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_size=args.model_size
    )
    
    if model_path:
        print(f"\n🎉 Success! Your model is ready at: {model_path}")
    else:
        print("\n⚠️  Training did not complete successfully.")
        sys.exit(1)

if __name__ == "__main__":
    main()
