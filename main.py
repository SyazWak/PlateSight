import streamlit as st
try:
    import cv2
    # Debug: Check OpenCV version and build info
    st.sidebar.write(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    st.error("Please ensure opencv-python-headless is properly installed.")
    # Additional debugging information
    import sys
    st.write(f"Python version: {sys.version}")
    st.write("Please try the following solutions:")
    st.write("1. Redeploy your app on Streamlit Cloud")
    st.write("2. Use the alternative requirements.txt file")
    st.write("3. Contact support if the issue persists")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error importing OpenCV: {e}")
    st.error("This might be a system-level issue with OpenCV.")
    st.stop()
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="inference")

# Optional imports for Roboflow functionality
try:
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

class PlateSightApp:
    def __init__(self):
        self.class_names = [
            "Anchovies", "Boiled-Egg", "Char-Kuey-Teow", "Chicken-Rendang", "Curry-Puff",
            "Fried-Chicken", "Fried-Egg", "Fried-Rice", "Hokkien-Mee", "Lo-Mein",
            "Mee-Rebus", "Mee-Siam", "Peanuts", "Rice", "Roti-Canai",
            "Sambal", "Slices-Cucumber"
        ]
        
        # Initialize session state
        if 'model_cache' not in st.session_state:
            st.session_state.model_cache = {}
    
    @st.cache_resource
    def load_model(_self, model_name):
        """Load YOLO model with caching"""
        try:
            if model_name not in st.session_state.model_cache:
                with st.spinner(f"Loading {model_name}..."):
                    model = YOLO(model_name)
                    st.session_state.model_cache[model_name] = model
                    st.success(f"✅ {model_name} loaded successfully!")
            return st.session_state.model_cache[model_name]
        except Exception as e:
            st.error(f"❌ Error loading {model_name}: {e}")
            # Fallback to YOLOv8n
            try:
                st.warning("Attempting to load fallback YOLOv8n model...")
                fallback_model = YOLO('yolov8n.pt')
                return fallback_model
            except Exception as e2:
                st.error(f"❌ Fallback model also failed: {e2}")
                return None
    
    def process_image_with_yolo(self, image, model, confidence):
        """Process image with YOLO model"""
        try:
            results = model(image, conf=confidence)
            annotated_image = results[0].plot()
            return annotated_image, results[0]
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return image, None
    
    def display_detection_info(self, results):
        """Display detection information in sidebar"""
        if results and results.boxes is not None:
            boxes = results.boxes
            num_detections = len(boxes)
            
            st.sidebar.subheader(f"🎯 Detections Found: {num_detections}")
            
            if num_detections > 0:
                # Group detections by class
                class_counts = {}
                for box in boxes:
                    class_id = int(box.cls.item())
                    if 0 <= class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                        confidence = box.conf.item()
                        
                        if class_name not in class_counts:
                            class_counts[class_name] = []
                        class_counts[class_name].append(confidence)
                
                # Display class summary
                st.sidebar.write("**Detected Items:**")
                for class_name, confidences in class_counts.items():
                    count = len(confidences)
                    avg_conf = np.mean(confidences)
                    st.sidebar.write(f"• {class_name}: {count}x (avg: {avg_conf:.2f})")
        else:
            st.sidebar.write("🔍 No detections found")
    
    def run(self):
        # Page configuration
        st.set_page_config(
            page_title="PlateSight - Malaysian Food Recognition",
            page_icon="🍽️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        .food-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 1rem 0;
        }
        .food-item {
            background: #f0f2f6;
            padding: 8px;
            border-radius: 8px;
            text-align: center;
            font-size: 12px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown('<h1 class="main-header">🍽️ PlateSight - Malaysian Food Recognition</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar configuration
        st.sidebar.header("⚙️ Configuration")
        
        # Model selection
        model_options = [
            "weights_YOLO_v11.pt",
            "weights_YOLO_v12.pt", 
            "weights_roboflow_3_0.pt"
        ]
        
        selected_model = st.sidebar.selectbox(
            "🤖 Select Model:",
            model_options,
            index=1,  # Default to YOLO v12
            help="Choose which model to use for detection"
        )
        
        # Confidence threshold
        confidence = st.sidebar.slider(
            "🎯 Confidence Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Higher values = fewer but more confident detections"
        )
        
        # Display supported food classes
        st.sidebar.subheader("🥘 Supported Food Classes")
        st.sidebar.markdown(
            '<div class="food-grid">' + 
            ''.join([f'<div class="food-item">{food}</div>' for food in self.class_names]) + 
            '</div>',
            unsafe_allow_html=True
        )
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📸 Upload and Analyze")
            
            # Input methods tabs
            tab1, tab2, tab3 = st.tabs(["📁 Image Upload", "🎬 Video Upload", "📹 Webcam"])
            
            with tab1:
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload an image of Malaysian food for analysis"
                )
                
                if uploaded_file is not None:
                    # Display original image
                    image = Image.open(uploaded_file)
                    
                    st.write("**Original Image:**")
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Process button
                    if st.button("🔍 Analyze Image", type="primary"):
                        # Load model
                        model = self.load_model(selected_model)
                        
                        if model is not None:
                            # Convert PIL to OpenCV format
                            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            
                            # Process image
                            with st.spinner("Processing image..."):
                                start_time = time.time()
                                annotated_image, results = self.process_image_with_yolo(
                                    cv_image, model, confidence
                                )
                                processing_time = time.time() - start_time
                            
                            # Convert back to RGB for display
                            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                            
                            # Display results
                            st.write("**Detection Results:**")
                            st.image(annotated_rgb, caption=f"Processed in {processing_time:.2f}s", 
                                   use_column_width=True)
                            
                            # Display detection info
                            self.display_detection_info(results)
            
            with tab2:
                uploaded_video = st.file_uploader(
                    "Choose a video file",
                    type=['mp4', 'avi', 'mov'],
                    help="Upload a video of Malaysian food for analysis"
                )
                
                if uploaded_video is not None:
                    # Save uploaded video to temporary file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    
                    # Display video with fixed size
                    st.write("**Original Video:**")
                    video_col1, video_col2, video_col3 = st.columns([1, 2, 1])
                    with video_col2:
                        # Fixed width container for video player
                        st.video(uploaded_video, start_time=0)
                    
                    # Video processing controls
                    st.write("**Analysis Controls:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        frame_skip = st.slider("Process every nth frame:", 1, 30, 5, 
                                             help="Lower values = more frames processed, Higher values = faster processing")
                    with col_b:
                        st.info("💡 **Tip:** Use lower frame skip values for detailed analysis, higher values for quick overview")
                    
                    # Control buttons
                    col_start, col_stop = st.columns(2)
                    with col_start:
                        start_analysis = st.button("🔍 Start Video Analysis", type="primary")
                    with col_stop:
                        stop_analysis = st.button("⏹️ Stop Analysis", type="secondary")
                    
                    # Initialize session state for video control
                    if 'video_playing' not in st.session_state:
                        st.session_state.video_playing = False
                    if 'video_stopped' not in st.session_state:
                        st.session_state.video_stopped = False
                    
                    if stop_analysis:
                        st.session_state.video_stopped = True
                        st.session_state.video_playing = False
                        st.warning("Video analysis stopped by user")
                    
                    if start_analysis and not st.session_state.video_playing:
                        st.session_state.video_playing = True
                        st.session_state.video_stopped = False
                        
                        model = self.load_model(selected_model)
                        
                        if model is not None:
                            # Process video
                            cap = cv2.VideoCapture(tfile.name)
                            
                            # Get video properties
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            st.write(f"**Video Info:** {total_frames} frames at {fps:.1f} FPS")
                            st.write(f"**Processing Settings:** Every {frame_skip} frames (processing {100/frame_skip:.1f}% of frames)")
                            
                            # Create placeholders for dynamic content
                            progress_bar = st.progress(0)
                            
                            # Create columns for frame display with fixed layout
                            st.write("**Analysis Results:**")
                            frame_col1, frame_col2, frame_col3 = st.columns([1, 3, 1])
                            with frame_col2:
                                frame_placeholder = st.empty()
                            
                            detection_placeholder = st.empty()
                            
                            frame_count = 0
                            processed_frames = 0
                            
                            try:
                                while cap.isOpened() and st.session_state.video_playing and not st.session_state.video_stopped:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    
                                    if frame_count % frame_skip == 0:
                                        # Process frame
                                        annotated_frame, results = self.process_image_with_yolo(
                                            frame, model, confidence
                                        )
                                        
                                        # Convert to RGB and display
                                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                        
                                        # Update frame display with fixed size
                                        frame_placeholder.image(frame_rgb, 
                                                              caption=f"Frame {frame_count}/{total_frames} | Processed: {processed_frames}",
                                                              width=600)  # Fixed width
                                        
                                        # Display detection info for current frame
                                        if results and results.boxes is not None:
                                            boxes = results.boxes
                                            num_detections = len(boxes)
                                            
                                            if num_detections > 0:
                                                # Group detections by class for current frame
                                                class_counts = {}
                                                for box in boxes:
                                                    class_id = int(box.cls.item())
                                                    if 0 <= class_id < len(self.class_names):
                                                        class_name = self.class_names[class_id]
                                                        confidence_score = box.conf.item()
                                                        
                                                        if class_name not in class_counts:
                                                            class_counts[class_name] = []
                                                        class_counts[class_name].append(confidence_score)
                                                
                                                # Display current frame detections
                                                detection_info = f"**🎯 Current Frame Detections: {num_detections}**\n\n"
                                                for class_name, confidences in class_counts.items():
                                                    count = len(confidences)
                                                    avg_conf = np.mean(confidences)
                                                    detection_info += f"• {class_name}: {count}x (avg: {avg_conf:.2f})\n"
                                                
                                                detection_placeholder.info(detection_info)
                                            else:
                                                detection_placeholder.info("🔍 No detections in current frame")
                                        else:
                                            detection_placeholder.info("🔍 No detections in current frame")
                                        
                                        processed_frames += 1
                                    
                                    frame_count += 1
                                    progress_bar.progress(frame_count / total_frames)
                                    
                                    # Check if user clicked stop (refresh session state)
                                    if st.session_state.video_stopped:
                                        break
                                
                                cap.release()
                                
                                if not st.session_state.video_stopped:
                                    st.success(f"✅ Video analysis completed! Processed {processed_frames} frames from {total_frames} total frames.")
                                
                            except Exception as e:
                                st.error(f"Error during video processing: {e}")
                            finally:
                                cap.release()
                                st.session_state.video_playing = False
                                # Clean up temp file
                                try:
                                    os.unlink(tfile.name)
                                except:
                                    pass
            
            with tab3:
                st.write("**Webcam Feature:**")
                st.info("📹 Webcam functionality requires running Streamlit locally. " +
                       "Use the camera input below or run the original tkinter version for real-time webcam.")
                
                # Camera input (Streamlit's built-in camera)
                camera_image = st.camera_input("Take a picture of Malaysian food")
                
                if camera_image is not None:
                    image = Image.open(camera_image)
                    
                    if st.button("🔍 Analyze Camera Image", type="primary"):
                        model = self.load_model(selected_model)
                        
                        if model is not None:
                            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            
                            with st.spinner("Processing camera image..."):
                                annotated_image, results = self.process_image_with_yolo(
                                    cv_image, model, confidence
                                )
                            
                            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                            st.image(annotated_rgb, caption="Camera Analysis Result", 
                                   use_column_width=True)
                            
                            self.display_detection_info(results)
        
        with col2:
            st.subheader("📊 Information")
            
            # Model info
            st.write("**Current Model:**")
            st.info(f"🤖 {selected_model}")
            
            # Performance info
            st.write("**Settings:**")
            st.write(f"• Confidence: {confidence:.1%}")
            
            # Tips
            st.subheader("💡 Tips for Best Results")
            st.write("""
            • **Good lighting** ensures better detection
            • **Clear view** of food items in frame
            • **Multiple dishes** can be detected in one image
            • **Lower confidence** finds more items
            • **Higher confidence** improves precision
            """)
            
            # Sample image test
            st.subheader("🧪 Quick Test")
            if os.path.exists("nasilemak.jpg"):
                if st.button("Test with Sample Image"):
                    model = self.load_model(selected_model)
                    if model is not None:
                        # Load and process sample image
                        sample_image = cv2.imread("nasilemak.jpg")
                        
                        with st.spinner("Processing sample image..."):
                            annotated_image, results = self.process_image_with_yolo(
                                sample_image, model, confidence
                            )
                        
                        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption="Sample Test Result")
                        
                        self.display_detection_info(results)
            else:
                st.write("Sample image not found")

def main():
    app = PlateSightApp()
    app.run()

if __name__ == "__main__":
    main()
