# ===================================================================================
#                           STREAMLIT APP: app.py (DEFINITIVE FINAL)
# ===================================================================================
import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from streamlit_image_comparison import image_comparison
from glob import glob
import json  # Added for loading mapping

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
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed') # We need this now
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Loading ---
@st.cache_resource
def load_model(model_path, device):
    """Loads the pre-trained model into memory."""
    model = ResNetUNet(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model

# --- Inference Function (works on .npy paths or uploaded PIL images) ---
def predict(model, image_input, device, is_uploaded_file=False):
    """Runs the full inference pipeline."""
    if is_uploaded_file:
        # Preprocessing for a single-channel uploaded image
        img = image_input.resize((240, 240)).convert('L')
        img_np = np.array(img, dtype=np.float32)
        max_val = np.max(img_np)
        if max_val > 0:
            img_np = img_np / max_val
        img_4_channel = np.stack([img_np] * 4, axis=-1)
        input_tensor = torch.from_numpy(img_4_channel).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        # Preprocessing for a 4-channel .npy file path
        img_np = np.load(image_input)
        input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask = (torch.sigmoid(logits) > 0.5).float()
        
    return pred_mask.squeeze(0).cpu().numpy()

# --- Load the winning model ---
BEST_MODEL_NAME = 'ResNetUNet'
model_path = os.path.join(MODELS_DIR, f'{BEST_MODEL_NAME}_best_model.pth')
model = load_model(model_path, DEVICE)

# --- Load the correct sample mapping ---
sample_png_files = sorted([f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.endswith('.png') and '_mask' not in f])

# Load the mapping from the JSON file created by your sample generation code
mapping_path = os.path.join(SAMPLE_IMAGES_DIR, 'sample_mapping.json')
if os.path.exists(mapping_path):
    with open(mapping_path, 'r') as f:
        sample_npy_map = json.load(f)
    st.success(f"✅ Loaded sample mapping for {len(sample_npy_map)} samples")
else:
    st.error("❌ Sample mapping file not found! Please run your sample creation code first.")
    sample_npy_map = {}

# ===================================================================================
#                                  STREAMLIT UI
# ===================================================================================
st.title("🧠 AI-Powered Brain MRI Tumor Segmentation")
st.markdown("""
This application demonstrates state-of-the-art deep learning for brain tumor segmentation using ResNet-UNet architecture.
Upload your own MRI scan or select from our curated samples to see the AI in action.
""")

st.sidebar.title("Controls")
st.sidebar.markdown("---")
selection_mode = st.sidebar.radio("Choose an option:", ("Select a Sample Image", "Upload Your Own Image"))

input_image = None
prediction_input = None
is_upload = False

if selection_mode == "Select a Sample Image":
    if sample_npy_map:
        selected_sample = st.sidebar.selectbox("Select sample:", sample_png_files)
        if selected_sample:
            input_image_path = os.path.join(SAMPLE_IMAGES_DIR, selected_sample)
            input_image = Image.open(input_image_path)
            # *** THE FIX: Use the corresponding .npy path for prediction ***
            npy_filename = sample_npy_map.get(selected_sample.replace('.png', '_image.npy').replace('_display',''))
            if npy_filename:
                prediction_input = os.path.join(SAMPLE_IMAGES_DIR, npy_filename)
                st.sidebar.success(f"✅ Using correct NPY file: {os.path.basename(prediction_input)}")
            else:
                st.sidebar.error(f"❌ No mapping found for {selected_sample}")
    else:
        st.sidebar.error("No sample mapping available. Please generate samples first.")
        
else:
    uploaded_file = st.sidebar.file_uploader("Upload a 2D MRI scan...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        prediction_input = input_image
        is_upload = True
        st.sidebar.warning("⚠️ Note: Model performance is optimized for 4-channel MRI data. Predictions on single-channel uploads serve as a limited demonstration.")

if input_image and prediction_input:
    st.markdown("---")
    st.subheader("Results")
    
    # Run prediction using the correct input type
    with st.spinner("🧠 AI is analyzing the brain scan..."):
        predicted_mask = predict(model, prediction_input, DEVICE, is_uploaded_file=is_upload)
        pred_mask_pil = Image.fromarray((predicted_mask.squeeze() * 255).astype(np.uint8))
    
    st.info("💡 Drag the slider to compare the original image with the AI's prediction.")
    image_comparison(img1=input_image, img2=pred_mask_pil, label1="Original Image", label2="Predicted Mask")
    
    st.markdown("### Detailed View")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(input_image, caption="Original Input Image", use_container_width=True)
    with col2:
        if selection_mode == "Select a Sample Image":
            gt_path = os.path.join(SAMPLE_IMAGES_DIR, selected_sample.replace('.png', '_mask.png'))
            if os.path.exists(gt_path):
                st.image(Image.open(gt_path), caption="Ground Truth Mask", use_container_width=True)
            else:
                st.info("Ground truth not available")
        else:
            st.info("Ground truth not available for uploaded images")
    with col3:
        st.image(pred_mask_pil, caption=f"AI Prediction ({BEST_MODEL_NAME})", use_container_width=True)

    # Add some metrics or additional info
    st.markdown("### Model Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Architecture", "ResNet-UNet")
    with col2:
        st.metric("Input Channels", "4 (Multi-modal MRI)")
    with col3:
        st.metric("Output", "Binary Mask")

else:
    st.info("👈 Please select or upload an image in the sidebar to begin.")
    
    # Show some example results when no image is selected
    st.markdown("### About This Application")
    st.markdown("""
    This brain tumor segmentation model was trained on multi-modal MRI data and achieves state-of-the-art performance:
    
    - **Architecture**: ResNet-UNet with skip connections
    - **Input**: 4-channel MRI data (T1, T1c, T2, FLAIR)
    - **Training Data**: BraTS dataset
    - **Performance**: Optimized for accurate tumor boundary detection
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
🔬 **Research Note**: This project benchmarks multiple architectures to find the optimal solution for brain tumor segmentation. 
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