# ===================================================================================
#                           STREAMLIT APP: app.py (CORRECTED)
# ===================================================================================
import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from streamlit_image_comparison import image_comparison
import json

# --- Add Project Root to Python Path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- Import our custom models ---
from src.models import ResNetUNet

# --- App Configuration ---
st.set_page_config(page_title="AI Brain Tumor Segmentation", page_icon="🧠", layout="wide")

# --- Define Paths ---
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SAMPLE_IMAGES_DIR = os.path.join(ROOT_DIR, 'streamlit_app', 'sample_images')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Loading ---
@st.cache_resource
def load_model(model_path, device):
    """Loads the pre-trained model into memory."""
    model = ResNetUNet(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model

# --- Inference Function ---
def predict(model, npy_array, device):
    """Runs the full inference pipeline on a 4-channel numpy array."""
    input_tensor = torch.from_numpy(npy_array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask = (torch.sigmoid(logits) > 0.5).float()
    return pred_mask.squeeze(0).cpu().numpy()

# --- Load the model ---
model_path = os.path.join(MODELS_DIR, 'ResNetUNet_best_model.pth')
model = load_model(model_path, DEVICE)

# --- Load the sample mapping ---
mapping_path = os.path.join(SAMPLE_IMAGES_DIR, 'sample_mapping.json')
if os.path.exists(mapping_path):
    with open(mapping_path, 'r') as f:
        sample_mapping = json.load(f)
else:
    sample_mapping = {}

# ===================================================================================
#                                    STREAMLIT UI
# ===================================================================================
st.title("🧠 AI-Powered Brain MRI Tumor Segmentation")

# --- UI Sidebar ---
st.sidebar.title("Controls")
selection_mode = st.sidebar.radio("Choose an option:", ("Select a Sample", "Upload Your Own .npy File"))

input_display_image = None
prediction_array = None
ground_truth_mask = None

if selection_mode == "Select a Sample":
    if sample_mapping:
        # Get list of sample names (the PNG keys)
        sample_names = list(sample_mapping.keys())
        selected_sample = st.sidebar.selectbox("Select sample:", sample_names)

        if selected_sample:
            # Get the corresponding .npy filenames from the mapping
            sample_info = sample_mapping[selected_sample]
            image_npy_filename = sample_info["image_npy"]  # e.g., "sample_1_image.npy"
            mask_npy_filename = sample_info["mask_npy"]    # e.g., "sample_1_mask.npy"
            
            # Load the actual .npy files
            image_npy_path = os.path.join(SAMPLE_IMAGES_DIR, image_npy_filename)
            mask_npy_path = os.path.join(SAMPLE_IMAGES_DIR, mask_npy_filename)
            
            if os.path.exists(image_npy_path) and os.path.exists(mask_npy_path):
                # Load the 4-channel image for prediction
                prediction_array = np.load(image_npy_path)
                
                # Load the ground truth mask
                ground_truth_mask = np.load(mask_npy_path)
                
                # Create display image from T1c channel (channel 0)
                display_slice = prediction_array[:, :, 0]  # T1c channel
                # Normalize to 0-255 range
                display_slice_norm = ((display_slice - display_slice.min()) / 
                                    (display_slice.max() - display_slice.min()) * 255)
                input_display_image = Image.fromarray(display_slice_norm.astype(np.uint8))
                
                st.sidebar.success(f"✅ Loaded {selected_sample}")
                st.sidebar.info(f"📁 Using: {image_npy_filename}")
                
            else:
                st.sidebar.error(f"❌ Data files not found for {selected_sample}")

    else:
        st.sidebar.error("❌ Sample mapping file not found.")

else:  # Upload Your Own .npy File
    uploaded_file = st.sidebar.file_uploader(
        "Upload a 4-channel MRI slice (.npy format)", 
        type=["npy"],
        help="The file should be a numpy array with shape (H, W, 4) containing 4 MRI channels"
    )
    
    if uploaded_file is not None:
        try:
            # Load the uploaded .npy file
            prediction_array = np.load(uploaded_file)
            
            # Validate the shape
            if len(prediction_array.shape) != 3 or prediction_array.shape[2] != 4:
                st.sidebar.error(f"❌ Invalid shape: {prediction_array.shape}. Expected (H, W, 4)")
                prediction_array = None
            else:
                # Create display image from T1c channel (channel 0)
                display_slice = prediction_array[:, :, 0]
                # Normalize to 0-255 range
                display_slice_norm = ((display_slice - display_slice.min()) / 
                                    (display_slice.max() - display_slice.min()) * 255)
                input_display_image = Image.fromarray(display_slice_norm.astype(np.uint8))
                
                st.sidebar.success("✅ File uploaded successfully!")
                st.sidebar.info(f"📊 Shape: {prediction_array.shape}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            prediction_array = None

# --- Main Page Display ---
if input_display_image is not None and prediction_array is not None:
    st.markdown("---")
    st.subheader("🔍 Analysis Results")

    with st.spinner("🧠 AI is analyzing the brain scan..."):
        predicted_mask_array = predict(model, prediction_array, DEVICE)
        pred_mask_pil = Image.fromarray((predicted_mask_array.squeeze() * 255).astype(np.uint8))

    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔄 Interactive Comparison")
        st.info("💡 Drag the slider to compare the original image with the AI's prediction.")
        image_comparison(
            img1=input_display_image, 
            img2=pred_mask_pil, 
            label1="Original MRI", 
            label2="AI Prediction"
        )
    
    with col2:
        # Show ground truth if available (for sample images)
        if ground_truth_mask is not None:
            st.markdown("### ✅ Ground Truth Comparison")
            ground_truth_pil = Image.fromarray(((ground_truth_mask > 0) * 255).astype(np.uint8))
            image_comparison(
                img1=pred_mask_pil,
                img2=ground_truth_pil,
                label1="AI Prediction",
                label2="Ground Truth"
            )
        else:
            st.markdown("### 📊 Prediction Statistics")
            tumor_pixels = np.sum(predicted_mask_array > 0.5)
            total_pixels = predicted_mask_array.size
            tumor_percentage = (tumor_pixels / total_pixels) * 100
            
            st.metric("Tumor Area", f"{tumor_percentage:.2f}%")
            st.metric("Tumor Pixels", f"{tumor_pixels:,}")
            st.metric("Image Size", f"{prediction_array.shape[0]}×{prediction_array.shape[1]}")

    # Add download button for predicted mask
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Convert prediction to downloadable format
        pred_mask_download = (predicted_mask_array.squeeze() * 255).astype(np.uint8)
        pred_mask_pil_download = Image.fromarray(pred_mask_download)
        
        # Convert PIL image to bytes for download
        import io
        img_buffer = io.BytesIO()
        pred_mask_pil_download.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Prediction",
            data=img_buffer.getvalue(),
            file_name="predicted_tumor_mask.png",
            mime="image/png"
        )

else:
    # Welcome message when no file is selected
    st.markdown("---")
    st.info("👈 **Get Started:** Please select a sample image or upload your own .npy file in the sidebar to begin analysis.")
    
    # Show some info about the app
    st.markdown("### 🔬 About This Application")
    st.markdown("""
    This AI-powered tool uses a **ResNet-UNet architecture** to automatically segment brain tumors from MRI scans.
    
    **Features:**
    - 🧠 Advanced deep learning model trained on medical imaging data
    - 🔄 Interactive image comparison with slider control
    - 📊 Detailed analysis statistics
    - 📥 Downloadable prediction results
    - ✅ Ground truth comparison for sample images
    
    **How to use:**
    1. Select a pre-loaded sample from the sidebar, or
    2. Upload your own 4-channel MRI data (.npy format)
    3. View the AI's tumor segmentation results
    4. Compare with ground truth (for samples) or download results
    """)

# --- Sidebar Information ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Model Information")
st.sidebar.info(f"""
**Architecture:** ResNet-UNet  
**Device:** {DEVICE.upper()}  
**Input:** 4-channel MRI (H×W×4)  
**Output:** Binary tumor mask
""")

st.sidebar.markdown("### 🔬 Research Note")
st.sidebar.info("""
This project benchmarks multiple architectures to find the optimal solution for brain tumor segmentation. 
The ResNet-UNet architecture demonstrated superior performance in our comparative analysis.
""")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Developed by Ahmad Nayfeh</strong></p>
    <p>🔗 <a href='https://github.com/Ahmad-Nayfeh'>GitHub</a> | 
    💼 <a href='https://www.linkedin.com/in/ahmad-nayfeh2000/'>LinkedIn</a> | 
    📧 ahmadnayfeh2000@gmail.com</p>
</div>
""", unsafe_allow_html=True)