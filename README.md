# AI-Powered MRI Brain Tumor Segmentation

> Automated brain tumor segmentation using deep learning to assist medical professionals in faster, more accurate diagnoses.

[![Live Demo](https://img.shields.io/badge/🚀-Live%20Demo-blue?style=for-the-badge)](https://mri-tumor-segmentation-nahmad.streamlit.app/)

![Dataset Overview](images/dataset.gif)

## Why This Matters

Brain tumor segmentation is critical for diagnosis and treatment planning, but manual analysis is time-consuming and inconsistent:

- ⏱️ **Manual process**: 30 minutes to several hours per scan
- 🎯 **AI solution**: Under 5 minutes with superior consistency
- 📊 **Performance**: Matches or exceeds human expert accuracy

## Features

- 🔄 **End-to-end pipeline** from preprocessing to deployment
- 🏗️ **Multiple architectures** - BaselineUNet, ResNetUNet, and TransUNet
- 🌐 **Interactive web app** for real-time segmentation
- 📋 **Comprehensive preprocessing** for 3D MRI to 2D conversion

## Live Demo

![Demo Screenshot](images/demo.png)

Try the interactive segmentation tool: **[Launch Demo](https://mri-tumor-segmentation-nahmad.streamlit.app/)**

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MRI-Tumor-Segmentation.git
cd MRI-Tumor-Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Prepare your data**: Place MRI scans (`.nii.gz` format) in `data/raw/`
2. **Run notebooks in order**:
   - `01_data_understanding_and_eda.ipynb` - Explore the dataset
   - `02_preprocessing_and_augmentation.ipynb` - Convert 3D to 2D slices
   - `03_model_training_and_benchmarking.ipynb` - Train models
   - `04_inference_and_app_preparation.ipynb` - Prepare for deployment

## Models

| Model | Description |
|-------|-------------|
| **BaselineUNet** | Standard U-Net architecture for biomedical segmentation |
| **ResNetUNet** | Enhanced U-Net with pre-trained ResNet backbone |
| **TransUNet** | State-of-the-art combining Transformer encoder with U-Net decoder |

## Results

![Training Results](images/loss_plot.png)

The **ResNetUNet** model achieved the highest performance with superior Dice similarity scores, demonstrating excellent segmentation accuracy across all tumor regions.

## Project Structure

```
├── data/
│   ├── raw/          # Your MRI data goes here
│   └── processed/    # Generated 2D slices
├── notebooks/        # Step-by-step Jupyter notebooks
├── src/
│   └── models.py     # Model architectures
├── streamlit_app/    # Web application
└── requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- NumPy, OpenCV, Matplotlib

See `requirements.txt` for complete list.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

⭐ **Found this helpful?** Give it a star and share with others working on medical AI!