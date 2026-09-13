import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Import the core logic from main.py
import main as bhranth

st.set_page_config(page_title="Bhranth Meter", page_icon="📝", layout="wide")

st.title("Bhranth Meter 📝")
st.markdown("### The world's most scientific\* psychological handwriting analyzer.")
st.markdown("\* *Note: Not actually scientific. 100% fake metrics. For entertainment purposes only.*")

mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    ["Auto (Detect Content)", "Doodle Index (Boredom)", "Signature Shake (Stability)", "Tremor (Hand Control)", "Doodle Pattern (Spatial Chaos)"]
)

uploaded_file = st.file_uploader("Upload an image of handwriting, doodles, signatures, or drawn lines...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the file into memory and convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("Could not read image file.")
    else:
        # Show uploaded image
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=400)
        
        # Preprocess exactly like main.py
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        st.write("---")
        st.subheader("Analysis Results:")
        
        with st.spinner("Analyzing psychological state..."):
            try:
                result_img = None
                metric_lines = []
                
                if mode == "Auto (Detect Content)":
                    result_img, metric_lines = bhranth.analyze_auto(gray, thresh)
                elif mode == "Doodle Index (Boredom)":
                    result_img, metric_lines = bhranth.analyze_doodle_index(gray, thresh)
                    # For individual modes, we might need to add the metrics banner manually or just display them as text
                elif mode == "Signature Shake (Stability)":
                    result_img, metric_lines = bhranth.analyze_signature_shake(gray, thresh)
                elif mode == "Tremor (Hand Control)":
                    result_img, metric_lines = bhranth.analyze_tremor(gray, thresh)
                elif mode == "Doodle Pattern (Spatial Chaos)":
                    result_img, metric_lines = bhranth.analyze_doodle_pattern(gray, thresh)
                
                # Display output
                if mode == "Auto (Detect Content)":
                    # Auto mode already overlays the banner
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Bhranth Analysis", use_container_width=True)
                else:
                    # For other modes, show the image and metrics below
                    # We can use main.py's banner function to keep the look consistent
                    final_img = bhranth._hstack_with_banner([result_img], metric_lines, banner_height=200)
                    st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="Bhranth Analysis", use_container_width=True)
                    
            except ValueError as e:
                st.error(f"Analysis Failed: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
