import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL.Image import Resampling
import cv2

# Define fruit classes
CLASSES = [
    "Bad Apple", "Good Apple", "Mixed Apple",
    "Bad Banana", "Good Banana", "Mixed Banana",
    "Bad Guava", "Good Guava", "Mixed Guava",
    "Mixed Lime", "Bad Lime", "Good Lime",
    "Good Orange", "Good Orange", "Mixed Orange",
    "Bad Pomegranate", "Good Pomegranate", "Mixed Pomegranate"
]

# Custom CSS for a polished look
st.markdown("""
    <style>
    .reportview-container {
        background: #F5F5F5;
        color: #333;
    }
    .title {
        font-size: 38px; 
        color: #1b5e20; 
        text-align: center; 
        font-weight: bold; 
        margin-top: 15px;
    }
    .subtitle {
        font-size: 22px; 
        color: #2e7d32;
        text-align: center;
        margin-bottom: 20px;
        font-style: italic;
    }
    .info-box {
        background-color: #E8F5E9; 
        padding: 10px; 
        border-radius: 5px; 
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .stButton>button {
        background-color: #5CB85C; 
        color: white; 
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45A049;
    }
    .stRadio [role='radiogroup']>label>div {
        font-weight: 600;
        color: #388E3C;
    }
    .confidence-bar {
        width: 100%; 
        background-color: #ccc; 
        border-radius: 4px; 
        margin-top: 10px; 
        margin-bottom: 10px;
    }
    .confidence-bar-fill {
        height: 20px; 
        background-color: #66BB6A; 
        border-radius: 4px;
    }
    .centered-text {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(tflite_model_path="model.tflite"):
    """Load and cache the TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0]['shape'][1]

def preprocess_image(image, image_size):
    """Preprocess image for model input."""
    image = image.convert("RGB").resize((image_size, image_size), Resampling.LANCZOS)
    return np.array(image, dtype=np.float32)[np.newaxis, ...]

def predict_quality(interpreter, image_array):
    """Predict fruit quality with confidence thresholds."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    confidences = interpreter.get_tensor(output_details[0]['index'])[0]
    
    max_pos = np.argmax(confidences)
    max_confidence = confidences[max_pos]

    if max_confidence > 0.75:
        result = CLASSES[max_pos]
    elif max_confidence > 0.35:
        result = f"Looks like {CLASSES[max_pos]}"
    else:
        result = "Couldn't identify!"
    return result, max_confidence

def get_emoji(result_text):
    """Return an emoji based on prediction result."""
    return "😊" if "Good" in result_text else "😐" if "Mixed" in result_text else "😞" if "Bad" in result_text or "Couldn't identify!" in result_text else "🤔"

def display_emoji(emoji):
    """Display a large centered emoji."""
    st.markdown(f"<p style='font-size:60px; text-align:center;'>{emoji}</p>", unsafe_allow_html=True)

def display_confidence_bar(confidence):
    """Display a custom-styled confidence bar."""
    bar_fill_width = int(confidence * 100)
    bar_html = f"""
        <div class="confidence-bar">
            <div class="confidence-bar-fill" style="width: {bar_fill_width}%;"></div>
        </div>
        <p class="centered-text"><strong>Confidence:</strong> {confidence:.2%}</p>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

def real_time_detection_in_browser(interpreter, image_size):
    """
    Capture webcam frames and display them directly in Streamlit,
    running inference on each frame without spawning a new OpenCV window.
    """
    if "run_webcam" not in st.session_state:
        st.session_state["run_webcam"] = False  # Default to off until started

    frame_window = st.empty()
    
    # Try different webcam indices (0, 1, 2) to find an available device
    cap = None
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            st.success(f"✅ Webcam found at index {index}")
            break
    if not cap or not cap.isOpened():
        st.error("🚫 Error: No webcam available or accessible. Check connections and permissions.")
        st.session_state["run_webcam"] = False
        return

    while st.session_state["run_webcam"]:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to grab frame! Stopping detection.")
            st.session_state["run_webcam"] = False
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        processed_image = preprocess_image(pil_image, image_size)
        result, confidence = predict_quality(interpreter, processed_image)

        overlay_text = f"{result} ({confidence:.2f})"
        annotated_frame = cv2.putText(
            frame,
            overlay_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        frame_window.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
    st.info("📷 Webcam released.")

def main():
    if "camera_error" not in st.session_state:
        st.session_state["camera_error"] = ""

    # **Title and Subtitle**
    st.markdown('<div class="title">Fruit Quality Predictor 🍎🍌</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze your fruits with AI precision!</div>', unsafe_allow_html=True)
    
    # **Information Box**
    st.markdown('<div class="info-box">Choose an option below to get started:</div>', unsafe_allow_html=True)

    # **Load model once**
    interpreter, image_size = load_model()

    # **Radio options**
    option = st.radio("", ["Upload Image", "Single Snapshot", "Real-Time Detection"])

    # **Option 1: Upload Image**
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            with st.spinner("Processing your image..."):
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Fruit", use_container_width=True)
                processed_image = preprocess_image(image, image_size)
                result, confidence = predict_quality(interpreter, processed_image)
                emoji = get_emoji(result)

                st.write(f"**Prediction:** {result}")
                display_confidence_bar(confidence)
                display_emoji(emoji)

    # **Option 2: Single Snapshot using Camera**
    elif option == "Single Snapshot":
        camera_image = st.camera_input("Snap a fruit photo")
        if camera_image:
            with st.spinner("Analyzing snapshot..."):
                image = Image.open(camera_image)
                st.image(image, caption="Captured Fruit", use_container_width=True)
                processed_image = preprocess_image(image, image_size)
                result, confidence = predict_quality(interpreter, processed_image)
                emoji = get_emoji(result)

                st.write(f"**Prediction:** {result}")
                display_confidence_bar(confidence)
                display_emoji(emoji)

    # **Option 3: Real-Time Detection**
    else:
        st.subheader("Real-Time Detection")
        st.markdown(
            '<div class="info-box">Click "Start Live Detection" to begin streaming your webcam feed.<br>'
            'Click "Stop Live Detection" to stop. Requires a local webcam.</div>',
            unsafe_allow_html=True
        )

        if st.session_state["camera_error"]:
            st.warning(st.session_state["camera_error"])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start Live Detection"):
                st.session_state["run_webcam"] = True
                real_time_detection_in_browser(interpreter, image_size)

        with col2:
            if st.button("Stop Live Detection"):
                st.session_state["run_webcam"] = False
                st.success("✅ Real-time detection stopped.")

if __name__ == "__main__":
    main()