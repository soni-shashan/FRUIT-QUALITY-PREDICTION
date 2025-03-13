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
    .title { font-size: 36px; color: #2E7D32; text-align: center; font-weight: bold; }
    .subtitle { font-size: 20px; color: #388E3C; text-align: center; }
    .info-box { background-color: #E8F5E9; padding: 10px; border-radius: 5px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stButton>button:hover { background-color: #45A049; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(tflite_model_path="model.tflite"):
    """Load and cache the TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0]['shape'][1]

def preprocess_image(image, image_size):
    """Preprocess image efficiently for model input."""
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

def real_time_detection(interpreter, image_size):
    """Run real-time detection in an OpenCV window."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("🚫 Error: Could not access webcam!")
        return

    cv2.namedWindow("Fruit Quality Live", cv2.WINDOW_NORMAL)
    st.info("Live detection started! Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to grab frame!")
            break

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        processed_image = preprocess_image(pil_image, image_size)
        result, confidence = predict_quality(interpreter, processed_image)

        # Overlay text
        cv2.putText(frame, f"{result} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Fruit Quality Live", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("✅ Real-time detection stopped.")

def main():
    # Session state initialization
    if "camera_error" not in st.session_state:
        st.session_state["camera_error"] = ""

    # UI Header
    st.markdown('<div class="title">Fruit Quality Predictor 🍎🍌</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze your fruits with AI precision!</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Choose an option below to get started:</div>', unsafe_allow_html=True)

    # Load model once
    interpreter, image_size = load_model()

    # Options
    option = st.radio("", ["Upload Image", "Single Snapshot", "Real-Time Detection"])

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
                st.write(f"**Confidence:** {confidence:.2%}")
                display_emoji(emoji)

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
                st.write(f"**Confidence:** {confidence:.2%}")
                display_emoji(emoji)

    else:
        st.subheader("Real-Time Detection")
        st.markdown('<div class="info-box">Click below to start live detection in a new window.<br>Press \'q\' to stop.</div>', unsafe_allow_html=True)
        
        if st.session_state["camera_error"]:
            st.warning(st.session_state["camera_error"])

        if st.button("Start Live Detection"):
            st.session_state["camera_error"] = ""
            real_time_detection(interpreter, image_size)

if __name__ == "__main__":
    main()