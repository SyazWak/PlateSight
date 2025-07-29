import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os
import threading
import warnings
import time
import queue
from concurrent.futures import ThreadPoolExecutor

# Suppress inference package warnings
warnings.filterwarnings("ignore", category=UserWarning, module="inference")

# Optional imports for Roboflow functionality
try:
    from inference.models.utils import get_roboflow_model
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Malaysian Food Recognition - YOLO Models")
        self.master.geometry("1200x800")
        self.stop_event = threading.Event()
        
        # Malaysian food class names
        self.class_names = [
            "Anchovies", "Boiled-Egg", "Char-Kuey-Teow", "Chicken-Rendang", "Curry-Puff",
            "Fried-Chicken", "Fried-Egg", "Fried-Rice", "Hokkien-Mee", "Lo-Mein",
            "Mee-Rebus", "Mee-Siam", "Peanuts", "Rice", "Roti-Canai",
            "Sambal", "Slices-Cucumber"
        ]
        
        # Roboflow configuration - Update these with your actual values
        self.roboflow_api_key = None  # Set your API key here
        self.roboflow_model_id = None  # Set your model ID here (e.g., "workspace/model-name/version")

        # Performance optimization variables
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to prevent lag
        self.result_queue = queue.Queue(maxsize=2)
        self.processing_executor = ThreadPoolExecutor(max_workers=2)
        self.last_processed_frame = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Performance settings
        self.skip_frames = 3  # Process every 3rd frame for better performance
        self.use_threading = True  # Enable threaded processing

        # set grid layout 1x2
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self.master, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="  🍽️ PlateSight",
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # Model Selection
        self.model_label = customtkinter.CTkLabel(self.navigation_frame, text="Model:")
        self.model_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.model_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=[
            "weights_roboflow_3_0.pt", "weights_YOLO_v11.pt", "weights_YOLO_v12.pt"
        ])
        self.model_menu.grid(row=2, column=0, padx=20, pady=10)

        # Confidence Slider
        self.confidence_label = customtkinter.CTkLabel(self.navigation_frame, text="Confidence:")
        self.confidence_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.confidence_slider = customtkinter.CTkSlider(self.navigation_frame, from_=0, to=1, command=self.update_confidence)
        self.confidence_slider.set(0.5)
        self.confidence_slider.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        self.confidence_value_label = customtkinter.CTkLabel(self.navigation_frame, text="50%")
        self.confidence_value_label.grid(row=6, column=0, padx=20, pady=(10,0), sticky="ew")

        # Performance Settings
        self.performance_label = customtkinter.CTkLabel(self.navigation_frame, text="Performance:")
        self.performance_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.fps_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=[
            "High Quality (Every Frame)", "Balanced (Every 3rd Frame)", "High FPS (Every 5th Frame)"
        ], command=self.update_performance_mode)
        self.fps_mode_menu.set("Balanced (Every 3rd Frame)")
        self.fps_mode_menu.grid(row=8, column=0, padx=20, pady=5)

        # FPS Display
        self.fps_label = customtkinter.CTkLabel(self.navigation_frame, text="FPS: --")
        self.fps_label.grid(row=9, column=0, padx=20, pady=(5, 10))

        # Input Buttons
        self.image_button = customtkinter.CTkButton(self.navigation_frame, text="📁 Select Image", command=self.select_image)
        self.image_button.grid(row=10, column=0, padx=20, pady=10)
        self.video_button = customtkinter.CTkButton(self.navigation_frame, text="🎬 Select Video", command=self.select_video)
        self.video_button.grid(row=11, column=0, padx=20, pady=10)
        self.webcam_button = customtkinter.CTkButton(self.navigation_frame, text="📹 Start Webcam", command=self.start_webcam)
        self.webcam_button.grid(row=12, column=0, padx=20, pady=10)
        self.stop_webcam_button = customtkinter.CTkButton(self.navigation_frame, text="⏹️ Stop Webcam", command=self.stop_webcam)
        self.stop_webcam_button.grid(row=13, column=0, padx=20, pady=10)
        self.test_button = customtkinter.CTkButton(self.navigation_frame, text="🧪 Test Display", command=self.test_display)
        self.test_button.grid(row=14, column=0, padx=20, pady=10)
        self.master.grid_rowconfigure(14, weight=1)

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self.master, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)
        self.home_frame.grid_rowconfigure(0, weight=1)
        self.home_frame.grid(row=0, column=1, sticky="nsew")

        self.display_label = customtkinter.CTkLabel(self.home_frame, text="Select an image, video, or start webcam to begin detection", width=800, height=600)
        self.display_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    def update_confidence(self, value):
        self.confidence_value_label.configure(text=f"{int(value * 100)}%")
        print(f"Confidence updated to: {value:.2f} ({int(value * 100)}%)")

    def update_performance_mode(self, mode):
        """Update performance settings based on selected mode"""
        if "Every Frame" in mode:
            self.skip_frames = 1
        elif "Every 3rd Frame" in mode:
            self.skip_frames = 3
        elif "Every 5th Frame" in mode:
            self.skip_frames = 5
        print(f"Performance mode changed: {mode} (skip_frames={self.skip_frames})")

    def update_fps_display(self):
        """Update FPS display"""
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        if elapsed >= 1.0:  # Update every second
            fps = self.fps_counter / elapsed
            self.fps_label.configure(text=f"FPS: {fps:.1f}")
            self.fps_counter = 0
            self.fps_start_time = current_time

    def select_image(self):
        self.stop_event.set()  # Stop any running video
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Run image processing in a separate thread
            self.image_thread = threading.Thread(target=self.process_image, args=(file_path,))
            self.image_thread.daemon = True
            self.image_thread.start()

    def load_model_safely(self, model_name):
        """Load YOLO or Roboflow model with error handling"""
        try:
            return self.load_yolo_model(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to a basic YOLO model
            try:
                print("Attempting to load default YOLOv8 model...")
                model = YOLO('yolov8n.pt')  # This will download if not present
                return {'model': model, 'type': 'yolo'}
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                return None

    def load_yolo_model(self, model_name):
        """Load YOLO model"""
        model = YOLO(model_name)
        return {'model': model, 'type': 'yolo'}

    def process_with_model(self, image, model_dict, confidence):
        """Process image with YOLO model"""
        if model_dict['type'] == 'yolo':
            return self.process_with_yolo(image, model_dict['model'], confidence)
        elif model_dict['type'] == 'roboflow':
            return self.process_with_roboflow(image, model_dict, confidence)
        else:
            return image

    def process_with_yolo(self, image, model, confidence):
        """Process image with YOLO model"""
        print(f"Processing with YOLO - Confidence: {confidence:.2f}")
        results = model(image, conf=confidence)
        return results[0].plot()

    def process_with_roboflow(self, image, model_dict, confidence):
        """Process image with Roboflow model"""
        if not ROBOFLOW_AVAILABLE:
            print("Roboflow SDK not available")
            return image
        try:
            client = model_dict['client']
            model_id = model_dict['model_id']
            print(f"Processing with Roboflow - Confidence: {confidence:.2f}")
            
            # Convert image to RGB format for Roboflow
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference via Roboflow API
            result = client.infer(rgb_image, model_id=model_id)
            
            # Draw predictions on the image
            annotated_image = rgb_image.copy()
            
            if 'predictions' in result and result['predictions']:
                for prediction in result['predictions']:
                    # Get bounding box coordinates
                    x_center = int(prediction['x'])
                    y_center = int(prediction['y'])
                    width = int(prediction['width'])
                    height = int(prediction['height'])
                    
                    # Calculate corners
                    x1 = x_center - width // 2
                    y1 = y_center - height // 2
                    x2 = x_center + width // 2
                    y2 = y_center + height // 2
                    
                    # Get confidence and class
                    conf = prediction.get('confidence', 0)
                    class_name = prediction.get('class', 'Object')
                    
                    # Only draw if confidence is above threshold
                    if conf >= confidence:
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(annotated_image, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert back to BGR for OpenCV
            return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error processing with Roboflow: {e}")
            return image

    def process_frame_async(self, frame, model_dict, confidence):
        """Process frame asynchronously"""
        try:
            return self.process_with_model(frame, model_dict, confidence)
        except Exception as e:
            print(f"Error in async processing: {e}")
            return frame

    def process_image(self, file_path):
        model_name = self.model_menu.get()
        confidence = self.confidence_slider.get()

        model_dict = self.load_model_safely(model_name)
        if model_dict is None:
            print("No valid model available")
            return

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image: {file_path}")
            return

        # Process with the appropriate model
        annotated_frame = self.process_with_model(image, model_dict, confidence)
        self.master.after(0, self._update_display, annotated_frame)

    def select_video(self):
        self.stop_event.set()  # Stop previous video processing
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            self.stop_event.clear()
            self.video_thread = threading.Thread(target=self.process_video, args=(file_path,))
            self.video_thread.daemon = True
            self.video_thread.start()

    def process_video(self, file_path):
        model_name = self.model_menu.get()
        model_dict = self.load_model_safely(model_name)
        
        if model_dict is None:
            print("No valid model available")
            return

        cap = cv2.VideoCapture(file_path)

        while cap.isOpened() and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # Get confidence value dynamically for each frame
            confidence = self.confidence_slider.get()
            
            # Process frame with the appropriate model
            annotated_frame = self.process_with_model(frame, model_dict, confidence)

            self.master.after(1, self._update_display, annotated_frame)
        
        cap.release()

    def _update_display(self, frame):
        # This function is called from the main thread
        try:
            # Convert BGR to RGB for PIL
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                im = Image.fromarray(frame)

            # Use fixed display dimensions to prevent shrinking
            display_width = 800
            display_height = 600

            # Calculate resize dimensions while maintaining aspect ratio
            img_width, img_height = im.size
            
            # Calculate scale factor
            scale_w = display_width / img_width
            scale_h = display_height / img_height
            scale = min(scale_w, scale_h)
            
            if scale > 0:  # Prevent division by zero
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                if new_width > 0 and new_height > 0:
                    im = im.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=im)

            # Update the display
            self.display_label.configure(image=photo, text="")
            self.display_label.image = photo  # Keep a reference
            
            # Update FPS counter
            self.fps_counter += 1
            self.update_fps_display()
            
        except Exception as e:
            print(f"Error updating display: {e}")
            import traceback
            traceback.print_exc()

    def start_webcam(self):
        self.stop_event.set()  # Stop any running video/webcam
        time.sleep(0.1)  # Give time for threads to stop
        self.stop_event.clear()
        
        # Reset FPS counter
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        self.webcam_thread = threading.Thread(target=self.process_webcam_optimized)
        self.webcam_thread.daemon = True
        self.webcam_thread.start()

    def stop_webcam(self):
        self.stop_event.set()

    def process_webcam_optimized(self):
        """Optimized webcam processing with higher frame rates"""
        try:
            model_name = self.model_menu.get()
            model_dict = self.load_model_safely(model_name)
            
            if model_dict is None:
                print("No valid model available for webcam")
                return

            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open camera")
                return
            
            # Optimize camera settings for higher FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 60)  # Try to set higher FPS
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
            
            # Get actual camera settings
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Webcam started with model: {model_name}")
            print(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps}fps")
            print(f"Processing every {self.skip_frames} frames")
            
            frame_count = 0
            last_inference_time = 0
            pending_future = None

            while cap.isOpened() and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                current_time = time.time()
                
                # Always show the latest frame (even if not processed)
                display_frame = frame.copy()
                
                # Check if we should process this frame
                should_process = (frame_count % self.skip_frames == 0)
                
                if should_process and self.use_threading:
                    # Check if previous processing is done
                    if pending_future is None or pending_future.done():
                        if pending_future and pending_future.done():
                            try:
                                # Get result from previous processing
                                processed_frame = pending_future.result()
                                self.last_processed_frame = processed_frame
                            except Exception as e:
                                print(f"Error getting async result: {e}")
                        
                        # Start new processing
                        confidence = self.confidence_slider.get()
                        pending_future = self.processing_executor.submit(
                            self.process_frame_async, frame.copy(), model_dict, confidence
                        )
                        
                elif should_process and not self.use_threading:
                    # Synchronous processing
                    confidence = self.confidence_slider.get()
                    processed_frame = self.process_with_model(frame, model_dict, confidence)
                    self.last_processed_frame = processed_frame
                
                # Use the last processed frame with detections if available
                if self.last_processed_frame is not None:
                    display_frame = self.last_processed_frame
                
                # Schedule GUI update on main thread
                self.master.after(0, self._update_display, display_frame)
                
                frame_count += 1
                
                # Dynamic frame rate control
                if actual_fps > 30:
                    time.sleep(0.01)  # Small delay for very high FPS cameras
                else:
                    time.sleep(0.005)  # Minimal delay for standard cameras
            
            cap.release()
            print(f"Webcam stopped. Processed {frame_count} frames.")
            
        except Exception as e:
            print(f"Error in webcam processing: {e}")
            import traceback
            traceback.print_exc()

    def test_display(self):
        """Test the display with a simple colored image"""
        try:
            import numpy as np
            # Create a test image (blue square with info)
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            test_image[:, :] = [50, 50, 200]  # Blue background
            
            # Add text
            cv2.putText(test_image, "GUIzaaboi - Malaysian Food Detection", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(test_image, "YOLO Models Ready", (150, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(test_image, f"Available Models: {len(self.model_menu.cget('values'))}", (100, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            self._update_display(test_image)
            print("Test display called")
        except Exception as e:
            print(f"Error in test display: {e}")

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
    
    root = customtkinter.CTk()
    app = App(root)
    root.mainloop()
