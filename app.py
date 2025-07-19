import streamlit as st
import cv2
import numpy as np
import easyocr
import os

# Streamlit page configuration
st.set_page_config(page_title="Text Removal App", layout="wide")

# Initialize EasyOCR reader
@st.cache_resource
def init_reader():
    try:
        return easyocr.Reader(['en'], gpu=False)  # Set gpu=True if GPU available
    except Exception as e:
        st.error(f"Error initializing EasyOCR: {e}")
        raise

reader = init_reader()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def detect_text(image):
    image_rgb = preprocess_image(image)
    try:
        results = reader.readtext(image_rgb, detail=1, contrast_ths=0.1, adjust_contrast=0.5)
    except Exception as e:
        st.error(f"Text detection failed: {e}")
        raise
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    text_sizes = []
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        text_sizes.append(max(width, height))
    
    avg_text_size = np.mean(text_sizes) if text_sizes else 10
    kernel_size = max(3, int(avg_text_size / 20))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask > 128).astype(np.uint8) * 255
    
    return mask, avg_text_size

def inpaint_text(image, mask, inpaint_radius, algorithm):
    if algorithm == "Navier-Stokes":
        return cv2.inpaint(image, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
    return cv2.inpaint(image, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

def main():
    st.title("Text Removal from Images")
    st.markdown(" Use Small Size files for the best results . \n Upload an image with text to remove it while preserving the background (e.g., forests, mountains, water).")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image having Small Size Memory (JPG/PNG)", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Failed to load image")
            return
        
        # User controls
        col1, col2 = st.columns(2)
        with col1:
            inpaint_radius = st.slider("Inpainting Radius", 5, 20, 10)
        with col2:
            algorithm = st.selectbox("Inpainting Algorithm", ["Navier-Stokes", "TELEA"])
        
        # Process image
        with st.spinner("Processing..."):
            mask, avg_text_size = detect_text(image)
            result = inpaint_text(image, mask, inpaint_radius, algorithm)
        
        # Display results
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        with col2:
            st.image(mask, caption="Text Mask", use_column_width=True)
        with col3:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Text Removed", use_column_width=True)
        
        # Download result
        _, img_buffer = cv2.imencode(".jpg", result)
        st.download_button(
            label="Download Result",
            data=img_buffer.tobytes(),
            file_name="output.jpg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error: {e}")
