# Particle Segmentation Tool

A comprehensive GUI-based application for particle image classification and segmentation using state-of-the-art deep learning models including SAM (Segment Anything Model), FastSAM, and custom CNN classifiers.

## 🌟 Features

- **Multi-Model Classification**: Support for ResNet50, ConvNext, MobileNet, EfficientNet, and DenseNet
- **Advanced Segmentation**: Integration with SAM and FastSAM for precise particle segmentation
- **Interactive GUI**: User-friendly PyQt6 interface for easy operation
- **Batch Processing**: Process multiple images with automatic saving and progress tracking
- **Mask Refinement**: Manual mask editing with Gaussian filtering and binary thresholding
- **Multi-Class Support**: 9 predefined particle classes with probability visualization
- **Export Capabilities**: Save processed images and labeled masks with metadata

## 🎯 Supported Particle Classes

1. **Blank-Oscillatory** - RGB: (255, 128, 0)
2. **Oscillatory-Sector OP** - RGB: (0, 255, 0)
3. **Sector-Oscillatory Sector OP** - RGB: (128, 255, 255)
4. **Sector-Sector** - RGB: (0, 0, 255)
5. **Stripy-Stripy** - RGB: (255, 0, 255)
6. **Wavy-Wavy** - RGB: (255, 128, 128)
7. **Unclassifiable** - RGB: (255, 255, 128)
8. **Metamict** - RGB: (192, 192, 192)
9. **Background** - RGB: (0, 0, 0)

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, or macOS 10.14+
- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU recommended (optional)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: 5GB+ free space for models and data

### Core Dependencies
```
PyQt6>=6.4.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.7.0
pandas>=2.0.0
Pillow>=8.3.0
scikit-image>=0.20.0
ultralytics>=8.0.0
segment-anything>=1.0
keras>=2.12.0
```

## 🚀 Installation

### Method 1: Conda Environment (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nazib/Particle-Segment
   cd particle-segmentation-tool
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate zirconapp
   ```

3. **Download required models**:
   ```bash
   # Download FastSAM model
   wget https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/weights/FastSAM-s.pt
   
   # Download SAM model (optional)
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```

4. **Set up model directories**:
   ```bash
   mkdir -p classification_models/Resnet50
   mkdir -p classification_models/ConvNext
   mkdir -p classification_models/MobileNet
   mkdir -p classification_models/EfficientNet
   mkdir -p classification_models/DenseNet
   mkdir -p binary_model
   ```

### Method 2: Pip Installation

1. **Create virtual environment**:
   ```bash
   python -m venv particle_env
   source particle_env/bin/activate  # On Windows: particle_env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install PyQt6 torch torchvision numpy opencv-python pandas Pillow scikit-image ultralytics segment-anything keras tensorflow
   ```

3. **Install additional requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Starting the Application

```bash
# Activate environment
conda activate zirconapp  # or source particle_env/bin/activate

# Run the application
python particle-segment_v.2.0.py
```
#### Create following folders:
1. **classification_models** (Download and Save all Classification models and save there from [here](https://drive.google.com/drive/folders/1oGrfC3Hkh2gXgCJR3DWN9m_zt_rpb_iw?usp=drive_link))
2. **binary_model** (Download and save binary model from [here:](https://drive.google.com/drive/folders/1oGrfC3Hkh2gXgCJR3DWN9m_zt_rpb_iw?usp=drive_link) 

### Basic Workflow

1. **Load Images**:
   - Click "Input Folder" to select your image directory
   - Supported formats: TIFF, JPG, PNG, JPEG
   - Click "Output Folder" to set save location

2. **Classification**:
   - Select classification model from dropdown (ResNet50, ConvNext, etc.)
   - Click "Classify" to predict particle class
   - View probability and class ID in the interface

3. **Mask Generation & Refinement**:
   - Click "Create/Refine Mask" for automatic segmentation
   - Check "Use SAM" for advanced segmentation with FastSAM
   - Adjust Gaussian Sigma (0-10) for smoothing
   - Adjust Binary Threshold (0-255) for edge detection

4. **Mask Selection & Application**:
   - Use dropdown to select from detected masks
   - Click "Apply Mask on Input Image" to preview
   - View results in the main display area

5. **Save Results**:
   - Click "Save Image" to export processed particle
   - Images saved to: `output_folder/Images/`
   - Masks saved to: `output_folder/Labels/`
   - Progress tracked in `SegmentationStatus.csv`

### Advanced Features

#### Multi-Particle Detection
- Automatic detection of multiple particles using binary classifier
- Warning dialog for user confirmation
- Recommendation to use SAM for separation

#### Mask Navigation
- Browse through detected masks using dropdown
- Real-time preview of selected masks
- Overlay visualization on original image

#### Batch Processing
- Process entire image directories
- Navigate with "Load Next Image" / "Load Previous Image"
- Automatic state persistence

## 📁 Project Structure

```
particle-segmentation-tool/
├── particle-segment_v.2.0.py      # Main application
├── classification.py              # Neural network models
├── utility.py                     # Image processing utilities
├── Particle-Segment.ui           # PyQt6 UI definition
├── environment.yml               # Conda environment
├── requirements.txt              # Pip requirements
├── classification_models/        # Trained model weights
│   ├── normalization_params.json
│   ├── Resnet50/
│   ├── ConvNext/
│   └── ...
├── binary_model/                 # Binary classifier
│   └── keras_model.h5
├── FastSAM-s.pt                 # FastSAM weights
└── README.md
```

## ⚙️ Configuration

### Model Configuration
Edit model paths in `particle-segment_v.2.0.py`:
```python
self.classification_model_path = "classification_models"
self.norm_param_file = "classification_models/normalization_params.json"
```

### Class Mapping
Modify class definitions in `classification.py`:
```python
color2class = {
    "your-class-name": [(R, G, B), class_id],
    # Add or modify classes as needed
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Segmentation Fault on Startup**:
   ```bash
   # Force software rendering
   export LIBGL_ALWAYS_SOFTWARE=1
   python particle-segment_v.2.0.py
   ```

2. **CUDA Out of Memory**:
   ```python
   # Add to code or reduce batch size
   torch.cuda.empty_cache()
   ```

3. **Qt Platform Plugin Issues**:
   ```bash
   # Try different platform
   export QT_QPA_PLATFORM=xcb  # or wayland, offscreen
   ```

4. **Missing Models**:
   - Ensure all model files are downloaded and placed correctly
   - Check file permissions and paths
   - Verify model compatibility with current PyTorch version

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA is properly installed
- **Memory Management**: Close unused applications when processing large images
- **Batch Size**: Reduce if experiencing memory issues
- **Image Size**: Consider resizing very large images (>4K) for faster processing

## 📊 Output Format

### Saved Images
- **Location**: `output_folder/Images/`
- **Format**: Original format preserved
- **Naming**: `original_name_maskN.extension`

### Label Masks
- **Location**: `output_folder/Labels/`
- **Format**: RGB PNG with class colors
- **Values**: Class-specific RGB values as defined in color mapping

### Metadata
- **File**: `SegmentationStatus.csv`
- **Columns**: FileIndex, FileName, ClassID, Probability, ClassName, MasksApplied
- **Format**: CSV with headers for easy analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Abdullah Nazib** - *Initial work* - [YourGitHub](https://github.com/nazib)

## 🙏 Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) by CASIA-IVA-Lab
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- PyQt6 team for the excellent GUI framework

## 📞 Support

For support, email your.email@example.com or create an issue on GitHub.

## 🔄 Version History

- **v2.0.0** - Major update with FastSAM integration and improved UI
- **v1.0.0** - Initial release with basic classification and segmentation

---

**Note**: This tool is designed for research purposes. Ensure proper validation for production use cases.
