# Text Removal from Images: Hackathon Solution

## Problem Statement
The challenge is to develop a system that detects and removes text from images while preserving complex backgrounds, such as natural scenes (forests, mountains, water). The solution must be robust, efficient, and user-friendly, suitable for deployment in a time-constrained hackathon environment (3 hours, ending ~1-2 PM IST, July 19, 2025).

## Approach
Our solution leverages computer vision and machine learning techniques to achieve high-quality text removal with minimal distortion to intricate backgrounds. The approach consists of three key stages:

1. **Text Detection**:
   - **Tool**: EasyOCR, a robust optical character recognition (OCR) library.
   - **Process**: 
     - Preprocess the input image using Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance text visibility on complex backgrounds.
     - Detect text regions using EasyOCR with tuned parameters (`contrast_ths=0.1`, `adjust_contrast=0.5`) for improved accuracy in low-contrast scenarios.
     - Generate a binary mask by drawing rectangles around detected text bounding boxes.

2. **Mask Refinement**:
   - **Dilation**: Apply adaptive dilation with a kernel size scaled by average text size (`max(3, int(avg_text_size / 20))`) to ensure complete text coverage.
   - **Smoothing**: Use Gaussian blur (`(5, 5)` kernel) to smooth mask edges, reducing inpainting artifacts.

3. **Text Removal (Inpainting)**:
   - **Tool**: OpenCV’s inpainting algorithms (`cv2.INPAINT_TELEA` and `cv2.INPAINT_NS`).
   - **Process**:
     - Compute inpainting with an adaptive radius (`max(5, min(15, int(avg_text_size / 10)))`) based on text size.
     - Compare `TELEA` (fast, smooth) and `Navier-Stokes` (texture-preserving) algorithms, selecting `Navier-Stokes` for larger text (>50 pixels) on complex backgrounds.
     - Display both results for user evaluation via a Streamlit interface.

4. **Deployment**:
   - **Platform**: Streamlit Community Cloud for quick, free deployment.
   - **Interface**: Interactive web app with file upload, adjustable inpainting radius (5-20), and algorithm selection (TELEA/Navier-Stokes).
   - **Output**: Users can download the text-removed image as `output.jpg`.

## Model Details
- **Text Detection**:
  - **Library**: EasyOCR (version 1.7.1).
  - **Configuration**: English language model, CPU-based (`gpu=False` for compatibility with Streamlit Cloud).
  - **Preprocessing**: CLAHE (clip limit 2.0, tile grid 8x8) to enhance contrast.
  - **Parameters**: `contrast_ths=0.1`, `adjust_contrast=0.5` for robust detection.

- **Masking**:
  - **Dilation**: Adaptive kernel size, 2 iterations.
  - **Smoothing**: Gaussian blur with `(5, 5)` kernel, thresholded at 128.

- **Inpainting**:
  - **Library**: OpenCV (version 4.12.0.88, headless).
  - **Algorithms**: `cv2.INPAINT_TELEA` (default for small text), `cv2.INPAINT_NS` (for complex backgrounds).
  - **Radius**: Adaptive, based on text size, capped at 5-15 pixels.

- **Dependencies**:
  - `streamlit==1.38.0`
  - `easyocr==1.7.1`
  - `opencv-python-headless==4.12.0.88`
  - `numpy==2.0.2`

## Implementation
- **Platform**: Streamlit web app deployed on Streamlit Community Cloud.
- **Source Code**: Available in the GitHub repository (`app.py`, `requirements.txt`).
- **Features**:
  - File uploader for JPG/PNG images.
  - Slider for inpainting radius and dropdown for algorithm selection.
  - Display of original image, text mask, and inpainted result.
  - Download button for the output image.
- **Deployment**:
  - Hosted at `<your-username>-text-removal-app.streamlit.app`.
  - Local testing: `pip install -r requirements.txt` and `streamlit run app.py`.

## Setup Instructions
1. **Clone Repository**:
   ```bash
   git clone https://github.com/hemanthkumarraya/text-removal-app
   cd text-removal-app
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Locally**:
   ```bash
   streamlit run app.py
   ```
4. **Access Deployed App**:
   - Visit `https://clove-technologies-text-removal-app.streamlit.app/`.
   - Upload an image, adjust parameters, and download the result.

## Results
- **Robustness**: Effectively removes text from complex backgrounds (e.g., forests, mountains, water) with minimal distortion.
- **User Experience**: Interactive Streamlit interface allows real-time parameter tuning.
- **Performance**: Processes images in ~2-5 seconds (CPU), suitable for hackathon demo.
- **Sample Outputs**: Input/output image pairs demonstrate high-quality text removal (included in repository).

## Future Improvements
- Integrate deep learning-based inpainting (e.g., LaMa) for enhanced background reconstruction (requires ~15 minutes setup, omitted due to hackathon time constraints).
- Add support for batch processing multiple images.
- Optimize for GPU to reduce processing time.

## Conclusion
This solution combines EasyOCR’s reliable text detection with OpenCV’s adaptive inpainting, deployed as a user-friendly Streamlit app. It addresses the hackathon problem efficiently, preserving complex backgrounds while providing an interactive interface for parameter tuning. The deployment on Streamlit Community Cloud ensures accessibility and ease of use for judges and users.
