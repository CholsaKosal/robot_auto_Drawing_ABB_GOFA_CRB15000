import tkinter as tk
from tkinter import messagebox, filedialog, ttk # Added ttk for progress bar
import os
import threading
import time
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import cv2 # <-- Added
import numpy as np # <-- Added (likely already implicitly used by cv2)
import math
from PIL import Image, ImageTk # <-- Added
# Consider adding tkinterdnd2 for drag-and-drop later if needed
# import tkinterdnd2

# --- Drawing Logic Imports ---
# (image_to_contours_internal, scale_point_to_a4, create_drawing_paths, calculate_distance) defined below

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants (Consolidated) ---
SCRIPT_DIR = os.getenv("SCRIPT_DIR", ".") # Default to current dir if not set
DATA_DIR = os.getenv("DATA_DIR", ".") # Default to current dir

# TMP_POSITION_FILE = os.path.join(DATA_DIR, "Tmp_position.txt") # Likely not needed for drawing
TMP_CAPTURE_PATH = os.path.join(DATA_DIR, "temp_capture.png") # For captured image
TMP_EDGE_OUTPUT_PATH = os.path.join(DATA_DIR, "temp_edges_{}.png") # For edge previews

REAL_ROBOT_HOST = '192.168.125.1'
REAL_ROBOT_PORT = 1025
SIMULATION_HOST = '127.0.0.1'
SIMULATION_PORT = 55000

# Drawing Specific Constants
FINAL_ROBOT_POSITION = (0, -350, 0) # Use X, Z, Y format (X, Depth, Y) - NOTE: Z is depth here
A4_WIDTH_MM = 180  # Drawing area width
A4_HEIGHT_MM = 217 # Drawing area height
PEN_DOWN_Z = -7   # Pen down position (depth)
PEN_UP_Z = 1.3 * PEN_DOWN_Z    # Pen up position (depth)
MIN_CONTOUR_LENGTH_PX = 10 # Minimum contour length in pixels to consider

# Threshold options
THRESHOLD_OPTIONS = [
    ("Option {}".format(i), i*10, i*20) for i in range(1, 8)
]

# Time estimation factor
TIME_ESTIMATE_FACTOR = 0.02 # seconds per command estimated

SIGNATURE_POINTS = ((0, ), (0, PEN_DOWN_Z, 0))

def create_signature_commands(points):
    """Converts raw signature points (X, Z, Y) into robot commands."""
    commands = []
    if not points:
        return commands

    # 1. Move to the start of the signature with Pen Up
    start_x, _, start_y = points[0] # Use X, Y from the first point
    commands.append((start_x, PEN_UP_Z, start_y)) # Ensure pen is up

    # 2. Add all signature points as commands (using their specified Z)
    for point in points:
        commands.append(point) # Add the point (x, z, y) directly

    # 3. Lift pen after the last point
    if commands:
        last_x, _, last_y = points[-1]
        commands.append((last_x, PEN_UP_Z, last_y)) # Lift pen at the end

    return commands


# --- Drawing Helper Function ---
def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x, y)."""
    if p1 is None or p2 is None: return float('inf')
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- Drawing Image Processing Functions ---
def image_to_contours_internal(image_path_or_array, threshold1, threshold2, save_edge_path=None):
    """
    Internal version: Convert image to contours using specific thresholds.
    Can accept a file path OR a pre-loaded cv2 image array.
    Does NOT print status messages.
    :param image_path_or_array: Path to the input image or numpy array (BGR or Grayscale).
    :param threshold1: Lower threshold for Canny edge detection.
    :param threshold2: Upper threshold for Canny edge detection.
    :param save_edge_path: Optional path to save the edge image for preview.
    :return: List of contours (pixel coordinates), image_width, image_height, or (None, 0, 0) on failure.
    """
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_path_or_array, np.ndarray):
        if len(image_path_or_array.shape) == 3: # BGR
            image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY)
        else: # Assuming already grayscale
            image = image_path_or_array
    else:
        logging.error("Invalid input type for image_to_contours_internal")
        return None, 0, 0

    if image is None:
        logging.error(f"Could not read or process image input.")
        return None, 0, 0

    image_height, image_width = image.shape[:2]
    if image_height == 0 or image_width == 0:
         logging.error("Invalid image dimensions.")
         return None, 0, 0

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1, threshold2)

    if save_edge_path:
        try:
            cv2.imwrite(save_edge_path, edges)
        except Exception as e:
            logging.error(f"Failed to save edge image to {save_edge_path}: {e}")

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.arcLength(c, closed=False) > MIN_CONTOUR_LENGTH_PX]

    contours_xy = []
    for contour in filtered_contours:
        points = contour.squeeze().tolist()
        if not isinstance(points, list) or not points: continue # Skip empty squeezes
        if isinstance(points[0], int): # Handle single point contour
            points = [points]
        contours_xy.append([(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]) # Ensure points are valid pairs

    # Filter out empty contours that might result from the above processing
    contours_xy = [c for c in contours_xy if c]

    return contours_xy, image_width, image_height


def scale_point_to_a4(point_xy, image_width, image_height, scale_factor):
    """ Scales and transforms a single (x, y) pixel coordinate to centered A4 (mm)."""
    x_pixel, y_pixel = point_xy
    center_x_pixel = image_width / 2
    center_y_pixel = image_height / 2
    x_centered_pixel = x_pixel - center_x_pixel
    y_centered_pixel = center_y_pixel - y_pixel # Invert y-axis
    x_mm = x_centered_pixel * scale_factor
    y_mm = y_centered_pixel * scale_factor
    return (x_mm, y_mm)

def create_drawing_paths(contours_xy, image_width, image_height, optimize_paths=True):
    """ Takes list of contours (pixel coordinates), scales them, creates drawing paths."""
    if not contours_xy or image_width <= 0 or image_height <= 0:
        return []

    scale_x = A4_WIDTH_MM / image_width
    scale_y = A4_HEIGHT_MM / image_height
    scale_factor = min(scale_x, scale_y)

    scaled_contours = []
    for contour in contours_xy:
        # Ensure contour is not empty before scaling
        if not contour: continue
        scaled_contour = [scale_point_to_a4(p, image_width, image_height, scale_factor) for p in contour]
        if len(scaled_contour) >= 2:
            scaled_contours.append(scaled_contour)
        elif len(scaled_contour) == 1 :
             # Handle single points - represent as a tiny segment back to itself?
             # This ensures it gets processed for pen down/up at least.
            scaled_contours.append([scaled_contour[0], scaled_contour[0]])


    if not scaled_contours:
        return []

    ordered_contours = []
    last_point = None # Keep track of the last point of the previously added contour
    if optimize_paths:
        remaining_contours = list(scaled_contours)
        # Find a starting contour (e.g., closest to origin, or just the first)
        # For simplicity, start with the first one if available.
        if remaining_contours:
             current_contour = remaining_contours.pop(0)
             ordered_contours.append(current_contour)
             last_point = current_contour[-1]

             while remaining_contours:
                 best_dist = float('inf')
                 best_idx = -1
                 best_reversed = False

                 for i, contour in enumerate(remaining_contours):
                     start_point = contour[0]
                     end_point = contour[-1]
                     dist_start = calculate_distance(last_point, start_point)
                     dist_end = calculate_distance(last_point, end_point)

                     if dist_start < best_dist:
                         best_dist = dist_start
                         best_idx = i
                         best_reversed = False
                     if dist_end < best_dist: # Check second condition independently
                         best_dist = dist_end
                         best_idx = i
                         best_reversed = True

                 if best_idx != -1:
                      next_contour = remaining_contours.pop(best_idx)
                      if best_reversed:
                          next_contour.reverse()
                      ordered_contours.append(next_contour)
                      last_point = next_contour[-1]
                 else:
                    # Should not happen if remaining_contours is not empty, but break just in case
                   logging.warning("Path optimization loop finished unexpectedly.")
                   break # Avoid infinite loop if something goes wrong
        scaled_contours = ordered_contours # Use the optimized order
        # logging.info(f"Optimized contour drawing order.") # Reduce noise
    else:
         # If not optimizing, just use the original order
         scaled_contours = [c for c in scaled_contours] # Ensure it's a list copy if needed


    robot_commands = []
    for contour in scaled_contours:
        if not contour: continue # Should not happen, but safe check
        start_point = contour[0]
        robot_commands.append((start_point[0], PEN_UP_Z, start_point[1])) # Move pen up to start X, Y
        robot_commands.append((start_point[0], PEN_DOWN_Z, start_point[1])) # Move pen down at start X, Y

        for i in range(len(contour) - 1):
            end_point = contour[i+1]
            # Avoid duplicate commands for single-point contours handled earlier
            if end_point != contour[i]:
                robot_commands.append((end_point[0], PEN_DOWN_Z, end_point[1])) # Draw to next point

        final_point = contour[-1]
        robot_commands.append((final_point[0], PEN_UP_Z, final_point[1])) # Lift pen at the end of contour

    return robot_commands


class RUNME_GUI:
    """Main GUI application for the Robotics System."""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Robotics Drawing GUI")
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Connection related variables
        self.connection_var = tk.StringVar(value="simulation")
        self.socket = None
        self.connected = False
        self.connection_established = False
        # self.positions = [] # Removed, not used for drawing 

        # Camera related variables
        self.cap = None
        self.camera_running = False
        self.camera_frame_label = None # Label to display camera feed 
        self.capture_button = None
        self.camera_back_button = None

        # Drawing process related
        self.current_image_path = None # Path to the image being processed
        self.threshold_options_data = {} # Store commands for each threshold choice
        self.selected_commands = None
        self.drawing_in_progress = False
        self.cancel_requested = False # *** NEW: Flag for cancellation ***
        self.progress_bar = None
        self.status_label = None
        self.cancel_button = None # *** NEW: Reference to cancel button ***
        self.reconnect_button = None # *** NEW: Reference to reconnect button ***

        self.last_drawing_status = {
            "total_commands": 0,
            "completed_commands": 0,
            "status": "None",  # e.g., "Completed", "Cancelled", "Connection Lost", "Protocol Error", "Failed to Resume"
            "error_message": ""
        }
        
        # Resume related variables
        self.resume_needed = False # *** NEW: Flag indicating connection was lost during drawing ***
        self.resume_commands = None # *** NEW: Store remaining commands ***
        self.resume_total_original_commands = 0 # *** NEW: Store original total for progress bar ***
        self.resume_start_index_global = 0 # *** NEW: Store the global index to resume from ***

        self.main_page()

    # --- Page Navigation ---
    def main_page(self):
        """Main application page."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Robotics Drawing System", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.main_frame, text="Setup Connection & Draw",
                  command=self.connection_setup_page, width=30).pack(pady=5)
        # tk.Button(self.main_frame, text="Camera Calibration", # Keep if needed 
        #           command=self.calibration_page, width=30).pack(pady=5) 
        tk.Button(self.main_frame, text="Exit",
                  command=self.on_window_close, width=30).pack(pady=5) # Call proper close handler

    def connection_setup_page(self):
        """Page for setting up robot connection."""
        # Simplified - remove radio buttons if only one target is needed often
        # Or keep as is 
        self.clear_frame()
        tk.Label(self.main_frame, text="Robot Connection Setup", font=("Arial", 16)).pack(pady=10)

        connection_frame = tk.Frame(self.main_frame)
        connection_frame.pack(pady=10)
        tk.Radiobutton(connection_frame, text=f"Simulation: {SIMULATION_HOST}:{SIMULATION_PORT}",
                       variable=self.connection_var, value="simulation").pack(anchor='w')
        tk.Radiobutton(connection_frame, text=f"Real Robot: {REAL_ROBOT_HOST}:{REAL_ROBOT_PORT}",
                       variable=self.connection_var, value="real").pack(anchor='w')

        # *** NEW: Conditionally show Connect or Reconnect & Resume button ***
        self.connect_button = tk.Button(self.main_frame, text="Connect", command=self.establish_connection, width=20)
        self.reconnect_button = tk.Button(self.main_frame, text="Reconnect & Resume", command=self.establish_connection, width=20) # Same command

        if self.resume_needed:
            self.reconnect_button.pack(pady=5)
            tk.Label(self.main_frame, text="Connection lost during last drawing. Reconnect to resume.", fg="orange").pack()
        else:
            self.connect_button.pack(pady=5)

        tk.Button(self.main_frame, text="Back", command=self.main_page, width=20).pack(pady=5) # Go back to main page

    def drawing_options_page(self):
        """Page shown after successful connection."""
        if not self.connection_established:
            messagebox.showerror("Connection Required", "Please establish connection first.")
            self.connection_setup_page()
            return

        self.clear_frame()
        tk.Label(self.main_frame, text="Robot Drawing Options", font=("Arial", 16)).pack(pady=10)
        conn_type = "Simulation" if self.connection_var.get() == "simulation" else "Real Robot"
        tk.Label(self.main_frame, text=f"Connected to: {conn_type}", fg="green").pack(pady=5)
        last_status = self.last_drawing_status["status"]
        if last_status not in ["None", "Completed"]:
            status_frame = tk.Frame(self.main_frame, relief=tk.RIDGE, borderwidth=2)
            status_frame.pack(pady=10, padx=10, fill='x')
            tk.Label(status_frame, text="Previous Drawing Status:", font=("Arial", 10, "bold")).pack(anchor='w')
            status_text = f"Status: {last_status}"
            if self.last_drawing_status["total_commands"] > 0:
                status_text += f" (Stopped at command {self.last_drawing_status['completed_commands'] + 1}" \
                                f" of {self.last_drawing_status['total_commands']})"
            tk.Label(status_frame, text=status_text).pack(anchor='w', padx=5)
            if self.last_drawing_status["error_message"]:
                tk.Label(status_frame, text=f"Details: {self.last_drawing_status['error_message']}", wraplength=400).pack(anchor='w', padx=5)


        tk.Button(self.main_frame, text="Capture Image to Draw",
                  command=self.capture_image_page, width=30).pack(pady=5) # Changed command
        tk.Button(self.main_frame, text="Input Image to Draw",
                  command=self.input_image_page, width=30).pack(pady=5) # Changed command
        tk.Button(self.main_frame, text="Disconnect",
                  command=self.close_and_return_main, width=30).pack(pady=5)

    # --- Capture Image Workflow ---
    def capture_image_page(self):
        """Opens camera view for capturing."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Camera View", font=("Arial", 16)).pack(pady=5)

        self.camera_frame_label = tk.Label(self.main_frame)
        self.camera_frame_label.pack(pady=10)

        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(pady=5)

        self.capture_button = tk.Button(button_frame, text="Capture (S)", command=self.capture_action)
        self.capture_button.pack(side=tk.LEFT, padx=5)
        self.camera_back_button = tk.Button(button_frame, text="Back", command=self.stop_camera_and_go_back)
        self.camera_back_button.pack(side=tk.LEFT, padx=5)

        # Bind 's' key
        self.window.bind('s', self.capture_action_event)
        self.window.bind('S', self.capture_action_event) # Also capital S

        self.start_camera_feed()

    def start_camera_feed(self):
        """Starts displaying the camera feed."""
        if self.camera_running: return # Already running

        try:
            self.cap = cv2.VideoCapture(0) # Use default camera 
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera.")
                self.stop_camera_and_go_back()
                return
            self.camera_running = True
            self._update_camera_frame() # Start the update loop
        except Exception as e:
             messagebox.showerror("Camera Error", f"Error initializing camera: {e}")
             self.stop_camera_and_go_back()


    def _update_camera_frame(self):
        """Internal method to continuously update the camera feed label."""
        if not self.camera_running or not self.cap:
             return

        ret, frame = self.cap.read()
        if ret:
            # Convert frame for Tkinter
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            # Resize image to fit nicely (optional)
            # aspect_ratio = pil_image.width / pil_image.height 
            # new_height = 300
            # new_width = int(aspect_ratio * new_height) 
            # pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS) 

            imgtk = ImageTk.PhotoImage(image=pil_image)

            if self.camera_frame_label: # Check if label still exists
                self.camera_frame_label.imgtk = imgtk
                self.camera_frame_label.configure(image=imgtk)
        else:
            logging.warning("Failed to grab frame from camera.")
            # Optionally try to reopen or show error after multiple failures

        # Schedule the next update
        if self.camera_running:
            self.window.after(30, self._update_camera_frame) # Update ~30fps

    def stop_camera_feed(self):
         """Stops the camera feed and releases resources."""
         self.camera_running = False # Signal the loop to stop
         time.sleep(0.1) # Give the loop a moment to exit
         if self.cap:
              self.cap.release()
              self.cap = None
         # cv2.destroyAllWindows() # Don't destroy all, might affect other CV windows if used 

    def stop_camera_and_go_back(self):
        """Stops camera and returns to drawing options page."""
        self.stop_camera_feed()
        self.window.unbind('s') # Unbind keys
        self.window.unbind('S')
        self.drawing_options_page() # Go back

    def capture_action_event(self, event=None):
         """Wrapper for key press event."""
         self.capture_action()

    def capture_action(self):
        """Captures the current frame and processes it."""
        if not self.camera_running or not self.cap:
            messagebox.showwarning("Capture Error", "Camera not running.")
            return

        ret, frame = self.cap.read()
        self.stop_camera_feed() # Stop feed after capture
        self.window.unbind('s') # Unbind keys
        self.window.unbind('S')


        if ret:
            try:
                # Ensure DATA_DIR exists
                os.makedirs(DATA_DIR, exist_ok=True)
                cv2.imwrite(TMP_CAPTURE_PATH, frame)
                logging.info(f"Image captured and saved to {TMP_CAPTURE_PATH}")
                self.current_image_path = TMP_CAPTURE_PATH
                # Proceed to threshold selection
                self.show_threshold_options(self.current_image_path)
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save captured image: {e}")
                self.drawing_options_page() # Go back on error
        else:
            messagebox.showerror("Capture Error", "Failed to capture frame from camera.")
            self.drawing_options_page() # Go back on error


    # --- Input Image Workflow ---
    def input_image_page(self):
        """Page for selecting an image file."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Input Image to Draw", font=("Arial", 16)).pack(pady=10)

        entry_frame = tk.Frame(self.main_frame)
        entry_frame.pack(pady=5, fill='x', padx=10)
        tk.Label(entry_frame, text="Image Path:").pack(side=tk.LEFT)
        self.image_path_var = tk.StringVar()
        path_entry = tk.Entry(entry_frame, textvariable=self.image_path_var, width=50)
        path_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
        tk.Button(entry_frame, text="Browse...", command=self.browse_image_file).pack(side=tk.LEFT)

        # Placeholder for drag-and-drop - requires tkinterdnd2
        # drop_target = tk.Label(self.main_frame, text="Or Drag and Drop Image Here", relief="ridge", height=5, width=60) 
        # drop_target.pack(pady=10)
        # drop_target.drop_target_register(DND_FILES) 
        # drop_target.dnd_bind('<<Drop>>', self.handle_drop) 

        tk.Button(self.main_frame, text="Process Image", command=self.process_input_image, width=20).pack(pady=10)
        tk.Button(self.main_frame, text="Back", command=self.drawing_options_page, width=20).pack(pady=10)

    def browse_image_file(self):
        """Opens file dialog to select an image."""
        filepath = filedialog.askopenfilename(
            title="Select Image to Draw", 
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")]
        )
        if filepath:
            self.image_path_var.set(filepath)

    # def handle_drop(self, event): # Requires tkinterdnd2 
    #     """Handles file drop event.""" 
    #     filepath = event.data.strip('{}') # Clean up path if needed 
    #     if os.path.isfile(filepath):
    #          self.image_path_var.set(filepath) 
    #     else: 
    #          messagebox.showwarning("Drop Error", f"Invalid file dropped: {filepath}") 


    def process_input_image(self):
        """Validates path and proceeds to threshold selection."""
        filepath = self.image_path_var.get()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror("Error", f"Invalid or non-existent file path:\n{filepath}")
            return
        self.current_image_path = filepath
        self.show_threshold_options(self.current_image_path)


    # --- Threshold Selection Workflow ---
    def show_threshold_options(self, image_path):
        """Processes image with different thresholds and shows options."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Select Drawing Style (Thresholds)", font=("Arial", 16)).pack(pady=10)

        self.threshold_options_data = {} # Clear previous results
        self.selected_threshold_option = tk.StringVar(value=None) # Variable for Radiobuttons
        self.preview_label = tk.Label(self.main_frame) # For showing edge previews
        self.preview_label.pack(pady=5)

        options_frame = tk.Frame(self.main_frame)
        options_frame.pack(pady=5)

        # Process each option in background to avoid freezing GUI
        loading_label = tk.Label(options_frame, text="Processing options...")
        loading_label.pack()
        self.window.update() # Show loading message 

        threading.Thread(target=self._process_threshold_options_thread, args=(image_path, options_frame, loading_label), daemon=True).start()


    def _process_threshold_options_thread(self, image_path, options_frame, loading_label):
        """Background thread to generate commands for each threshold option."""
        results = {}
        preview_paths = {} # Store paths to preview images

        for i, (label, t1, t2) in enumerate(THRESHOLD_OPTIONS):
            logging.info(f"Processing option: {label} (t1={t1}, t2={t2})")
            preview_path = TMP_EDGE_OUTPUT_PATH.format(i) # Unique path for preview
            contours_xy, w, h = image_to_contours_internal(image_path, t1, t2, save_edge_path=preview_path)

            if contours_xy is None or w == 0 or h == 0:
                 logging.warning(f"Failed to process contours for option {label}")
                 results[label] = None # Indicate failure 
                 preview_paths[label] = None
                 continue

            commands = create_drawing_paths(contours_xy, w, h, optimize_paths=True)
            if commands:
                num_commands = len(commands)
                est_time_sec = num_commands * TIME_ESTIMATE_FACTOR
                est_time_min = est_time_sec / 60
                results[label] = {
                    "commands": commands, 
                    "count": num_commands,
                    "time_str": f"{est_time_min:.1f} min",
                    "t1": t1,
                    "t2": t2
                }
                preview_paths[label] = preview_path if os.path.exists(preview_path) else None
            else:
                 results[label] = None # No commands generated
                 preview_paths[label] = None
                 logging.warning(f"No commands generated for option {label}")

        # Update GUI from the main thread
        self.window.after(0, lambda: self._display_threshold_options(options_frame, loading_label, results, preview_paths))


    def _display_threshold_options(self, options_frame, loading_label, results, preview_paths):
         """Updates the GUI with the processed threshold options."""
         loading_label.destroy() # Remove loading message

         self.threshold_options_data = results # Store results
         self.edge_preview_paths = preview_paths # Store preview paths

         default_selected = False
         for i, (label, t1, t2) in enumerate(THRESHOLD_OPTIONS):
             option_data = results.get(label)
             if option_data:
                 count = option_data["count"]
                 time_str = option_data["time_str"]
                 radio_text = f"{label} (t1={t1}, t2={t2}) - Cmds: {count}, Est: {time_str}"
                 rb = tk.Radiobutton(
                    options_frame, 
                 text=radio_text,
                 variable=self.selected_threshold_option,
                 value=label,
                 command=lambda l=label: self.show_edge_preview(l) # Show preview on select
                 )
                 rb.pack(anchor='w')
                 # Select the first valid option by default
                 if not default_selected:
                      self.selected_threshold_option.set(label)
                      self.show_edge_preview(label) # Show its preview
                      default_selected = True
             else:
                 # Option failed or produced no commands
                 tk.Label(options_frame, text=f"{label} (t1={t1}, t2={t2}) - No drawing generated", fg="gray").pack(anchor='w')

         # Add Confirm and Back buttons below the options
         button_frame = tk.Frame(self.main_frame)
         button_frame.pack(pady=10)
         tk.Button(button_frame, text="Confirm and Draw", command=self.confirm_and_start_drawing, width=20).pack(side=tk.LEFT, padx=5)
         # Back button should go back to the drawing options page (Capture/Input)
         tk.Button(button_frame, text="Back", command=self.drawing_options_page, width=20).pack(side=tk.LEFT, padx=5)

    def show_edge_preview(self, option_label):
         """Displays the edge preview image for the selected option."""
         preview_path = self.edge_preview_paths.get(option_label)
         if preview_path and os.path.exists(preview_path):
              try:
                   img = Image.open(preview_path)
                   # Resize for display
                   img.thumbnail((300, 300)) # Max width/height 300px
                   imgtk = ImageTk.PhotoImage(image=img)
                   self.preview_label.imgtk = imgtk
                   self.preview_label.configure(image=imgtk)
              except Exception as e:
                   logging.error(f"Error loading preview image {preview_path}: {e}")
                   self.preview_label.configure(image=None, text="Preview error") # Clear preview
         else:
              self.preview_label.configure(image=None, text="No Preview") # Clear preview


    def confirm_and_start_drawing(self):
        """Gets selected commands and starts the drawing process."""
        selected_label = self.selected_threshold_option.get()
        if not selected_label:
            messagebox.showwarning("Selection Needed", "Please select a drawing style option.")
            return

        option_data = self.threshold_options_data.get(selected_label)
        if not option_data or not option_data.get("commands"):
             messagebox.showerror("Error", "Selected option has no drawing commands.")
             return

        self.selected_commands = option_data["commands"]

        # Start drawing in a background thread
        if not self.drawing_in_progress:
             self.drawing_in_progress = True
             self.cancel_requested = False # Ensure cancel flag is reset
             self.resume_needed = False # Reset resume flag
             # *** Pass the full command list including signature ***
             full_command_list = self.selected_commands + create_signature_commands(SIGNATURE_POINTS)
             threading.Thread(target=self.run_drawing_loop, args=(full_command_list,), daemon=True).start()
             self.show_drawing_progress_page(len(full_command_list)) # Show progress UI with total commands
        else:
            messagebox.showwarning("Busy", "Drawing already in progress.")


    # --- Drawing Execution Workflow ---
    def show_drawing_progress_page(self, total_commands, current_progress=0, status_message="Starting..."):
         """Displays the progress bar and status during drawing."""
         self.clear_frame()
         tk.Label(self.main_frame, text="Drawing in Progress...", font=("Arial", 16)).pack(pady=10)

         self.status_label = tk.Label(self.main_frame, text=status_message)
         self.status_label.pack(pady=5)

         self.progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", length=300, mode="determinate", maximum=total_commands, value=current_progress)
         self.progress_bar.pack(pady=10)

         # *** NEW: Add Cancel Button ***
         self.cancel_button = tk.Button(self.main_frame, text="Cancel Drawing", command=self.request_cancel_drawing)
         self.cancel_button.pack(pady=5)


    def update_drawing_status(self, current_command_index, total_commands, message=""):
        """Callback to update progress bar and status label from drawing thread."""
        if self.progress_bar and self.progress_bar.winfo_exists():
            self.progress_bar['value'] = current_command_index
        if self.status_label and self.status_label.winfo_exists():
            status_text = f"Sent {current_command_index} / {total_commands} commands"
            if message:
                 status_text += f" ({message})"
            self.status_label.config(text=status_text)
            # self.window.update_idletasks() # Force update if needed, but 'after' usually handles it 

    def request_cancel_drawing(self):
        """Sets the cancellation flag when the Cancel button is pressed."""
        if self.drawing_in_progress:
            logging.info("Cancel requested by user.")
            self.cancel_requested = True
            if self.cancel_button and self.cancel_button.winfo_exists():
                self.cancel_button.config(text="Cancelling...", state=tk.DISABLED)
            if self.status_label and self.status_label.winfo_exists():
                self.status_label.config(text="Cancellation requested...")

    def _send_final_position_and_cleanup(self, success_message, failure_message):
        """Sends the robot to the final position and cleans up state. Runs in drawing thread."""
        logging.info("Attempting to move robot to final position.")
        final_x, final_z, final_y = FINAL_ROBOT_POSITION
        command_str_final = f"{final_x:.3f},{final_z:.3f},{final_y:.3f}" # Format final command 

        move_ok = False
        if self.connected and self.socket:
            if self.send_message_internal(command_str_final):
                response_r_final = self.receive_message_internal(timeout=20.0)
                if response_r_final == "R":
                    response_d_final = self.receive_message_internal(timeout=30.0) # Longer timeout for final move
                    # if response_d_final == "D":
                    #     logging.info("Robot reached final position.") #
                    #     move_ok = True
                    # else:
                    #     logging.error(f"Robot didn't confirm final move completion (D), got '{response_d_final}'") #
                else:
                    logging.error(f"Robot didn't confirm final move receipt (R), got '{response_r_final}'") #
            else:
                logging.error("Failed to send final position command.") #

        # Update GUI status based on move success/failure and original reason
        final_status = ""
        if move_ok:
            final_status = f"{success_message} Robot at final position."
        else:
            final_status = f"{failure_message} Failed to reach final position."

        self.last_drawing_status["status"] = success_message # Use the original reason (Completed, Cancelled, etc.)
        self.last_drawing_status["error_message"] = "" if move_ok else "Failed to reach final position."

        self.window.after(0, lambda fs=final_status: self.update_final_status(fs))

        # --- Final Cleanup ---
        self.drawing_in_progress = False
        self.selected_commands = None
        self.cancel_requested = False
        # Reset resume state ONLY if the process finished (successfully or cancelled), not on disconnect
        if not self.resume_needed: # Don't clear resume state if we dropped connection
            self.resume_commands = None
            self.resume_total_original_commands = 0
            self.resume_start_index_global = 0

        # Go back to the drawing options page after a short delay
        self.window.after(2000, self.drawing_options_page) # Wait 2s before going back 

    def update_final_status(self, message):
        """Updates the status label safely from the main thread."""
        if self.status_label and self.status_label.winfo_exists():
            self.status_label.config(text=message)
        if self.cancel_button and self.cancel_button.winfo_exists():
            self.cancel_button.pack_forget() # Remove cancel button

    def run_drawing_loop(self, commands_to_send: List[Tuple], start_index=0):
        """Sends drawing commands sequentially (RUNS IN THREAD). Handles cancel and resume."""
        total_commands = len(commands_to_send) + start_index # Total original commands for progress bar max
        current_command_global_index = start_index # Start from the correct global index
        commands_processed_in_this_run = 0

        # If resuming, ensure progress page reflects original total and current progress
        if start_index > 0:
            self.window.after(0, lambda: self.show_drawing_progress_page(total_commands, current_command_global_index, "Resuming drawing..."))
            self.window.after(0, lambda: self.update_drawing_status(current_command_global_index, total_commands, "Resuming..."))
        else:
            self.window.after(0, lambda: self.update_drawing_status(0, total_commands, "Starting..."))

        try:
            # Iterate through the commands *starting from the correct index*
            for i, (x, z, y) in enumerate(commands_to_send[start_index:], start=start_index):
                current_command_global_index = i + 1 # Overall progress index (1-based)

                # *** NEW: Check for cancellation before sending ***
                if self.cancel_requested:
                    logging.info(f"Cancellation detected at command {current_command_global_index}.")
                    self.window.after(0, lambda idx=i: self.update_drawing_status(idx, total_commands, "Cancelling..."))
                    self._send_final_position_and_cleanup("Drawing Cancelled.", "Drawing Cancelled.")
                    return # Exit the thread

                # Format command
                command_str = f"{x:.2f},{z},{y:.2f}" # Format for robot 
                logging.debug(f"Sending command {current_command_global_index}/{total_commands}: {command_str}")

                # --- Robot Communication Protocol ---
                # 1. Send Command
                if not self.send_message_internal(command_str): # If send fails...
                    # *** NEW: Handle connection loss ***
                    logging.error(f"Connection lost while sending command {current_command_global_index}. Preparing to resume.")
                    self.resume_needed = True
                    # Save state relative to the *original full list*
                    self.resume_commands = commands_to_send # Keep the full list
                    self.resume_start_index_global = i # Save the index of the command that failed (0-based)
                    self.resume_total_original_commands = total_commands
                    
                    self.last_drawing_status["total_commands"] = total_commands
                    self.last_drawing_status["completed_commands"] = i # Command i failed
                    self.last_drawing_status["status"] = "Connection Lost"
                    self.last_drawing_status["error_message"] = f"Lost connection before sending command {i+1}"
                    
                    self.window.after(0, lambda idx=i: self.update_drawing_status(idx, total_commands, "Connection Lost!"))
                    self.window.after(1000, self.connection_setup_page) # Go to connection page to allow reconnect
                    self.drawing_in_progress = False # Allow reconnect button to work
                    return # Exit thread

                # 2. Wait for Receipt 'R'
                response_r = self.receive_message_internal(timeout=20.0) # If receive fails...
                if response_r is None: # Check for None indicating socket error/timeout
                    # *** NEW: Handle connection loss ***
                    logging.error(f"Connection lost while waiting for 'R' after command {current_command_global_index}. Preparing to resume.")
                    self.resume_needed = True
                    self.resume_commands = commands_to_send
                    self.resume_start_index_global = i # Resume from the command that wasn't fully confirmed
                    self.resume_total_original_commands = total_commands
                    self.window.after(0, lambda idx=i: self.update_drawing_status(idx, total_commands, "Connection Lost! (No 'R')"))
                    self.window.after(1000, self.connection_setup_page)
                    self.drawing_in_progress = False
                    return # Exit thread
                elif response_r != "R":
                    error_msg = f"Robot did not confirm receipt (R) for command {current_command_global_index}, got '{response_r}'."
                    logging.error(error_msg + " Preparing to resume.") # Changed log message
                    # *** NEW: Prepare for resume on 'R' error ***
                    self.resume_needed = True
                    self.resume_commands = commands_to_send
                    self.resume_start_index_global = i # Resume from the command that failed confirmation
                    self.resume_total_original_commands = total_commands
                    # Update last status
                    self.last_drawing_status["total_commands"] = total_commands
                    self.last_drawing_status["completed_commands"] = i
                    self.last_drawing_status["status"] = "Protocol Error (R)"
                    self.last_drawing_status["error_message"] = error_msg
                    # *** End NEW ***
                    self.window.after(0, lambda idx=i, r=response_r: self.update_drawing_status(idx, total_commands, f"Error: No 'R' (Got {r}). Reconnect to resume."))
                    # *** NEW: Go to connection page instead of cleanup ***
                    self.window.after(1000, self.connection_setup_page)
                    self.drawing_in_progress = False
                    return # Exit thread

                # 3. Wait for Done 'D'
                response_d = self.receive_message_internal(timeout=30.0) # Longer timeout for move completion 
                if response_d is None: # Check for None indicating socket error/timeout
                    # *** NEW: Handle connection loss ***
                    logging.error(f"Connection lost while waiting for 'D' after command {current_command_global_index}. Preparing to resume.")
                    self.resume_needed = True
                    self.resume_commands = commands_to_send
                    # Resume from the *next* command since this one completed movement but confirmation failed
                    self.resume_start_index_global = i + 1
                    self.resume_total_original_commands = total_commands
                    self.window.after(0, lambda idx=i: self.update_drawing_status(idx + 1, total_commands, "Connection Lost! (No 'D')")) # Show progress for completed command
                    self.window.after(1000, self.connection_setup_page)
                    self.drawing_in_progress = False
                    return # Exit thread
                # elif response_d != "D":
                #     error_msg = f"Robot did not confirm completion (D) for command {current_command_global_index}, got '{response_d}'."
                #     logging.error(error_msg + " Preparing to resume.") # Changed log message
                #     # *** NEW: Prepare for resume on 'D' error ***
                #     self.resume_needed = True
                #     self.resume_commands = commands_to_send
                #     # Resume from the *next* command since 'R' was received, but 'D' failed
                #     self.resume_start_index_global = i + 1
                #     self.resume_total_original_commands = total_commands
                #     # Update last status
                #     self.last_drawing_status["total_commands"] = total_commands
                #     self.last_drawing_status["completed_commands"] = i + 1 # Command i movement likely completed
                #     self.last_drawing_status["status"] = "Protocol Error (D)"
                #     self.last_drawing_status["error_message"] = error_msg
                #     # *** End NEW ***
                #     self.window.after(0, lambda idx=i, d=response_d: self.update_drawing_status(idx + 1, total_commands, f"Error: No 'D' (Got {d}). Reconnect to resume."))
                #     # *** NEW: Go to connection page instead of cleanup ***
                #     self.window.after(1000, self.connection_setup_page)
                #     self.drawing_in_progress = False
                #     return # Exit thread

                commands_processed_in_this_run += 1
                # Update GUI progress
                self.window.after(0, lambda idx=current_command_global_index: self.update_drawing_status(idx, total_commands)) # 

            # If the loop completes without cancellation or error
            logging.info("All drawing commands sent successfully.")
            self.window.after(0, lambda: self.update_drawing_status(total_commands, total_commands, "Drawing Complete."))
            self._send_final_position_and_cleanup("Drawing Complete.", "Drawing Complete.")

        except Exception as e:
            logging.error(f"Unexpected error during drawing process: {e}", exc_info=True) # 
            # Attempt to update status, but might fail if GUI is gone
            try:
                self.window.after(0, lambda idx=current_command_global_index: self.update_drawing_status(idx, total_commands, f"Runtime Error: {e}")) # 
            except tk.TclError:
                logging.error("GUI already closed during error handling.")
            # Don't try to move robot here, connection state unknown
            self.drawing_in_progress = False
            self.cancel_requested = False
            # Keep resume state in case it was a connection error leading to this exception
            # self.window.after(2000, self.drawing_options_page) # Don't automatically go back on unexpected error


    # --- Internal Socket Methods (without GUI popups) ---
    def send_message_internal(self, message: str) -> bool:
        """ Sends message without triggering GUI popups on error. Returns success status. """
        if not self.connected or not self.socket: return False
        try:
            self.socket.sendall(message.encode('utf-8'))
            logging.debug(f"Sent (internal): {message}")
            return True
        except (socket.error, ConnectionResetError, BrokenPipeError, socket.timeout) as e: # Added BrokenPipeError
            logging.error(f"Send error (internal): {e}")
            self.handle_connection_loss() # Use centralized handler
            return False

    def receive_message_internal(self, timeout=20.0) -> Optional[str]:
         """ Receives message without triggering GUI popups on error. Returns message or None. """
         if not self.connected or not self.socket: return None
         try:
             self.socket.settimeout(timeout)
             data = self.socket.recv(1024)
             self.socket.settimeout(None) # Reset timeout
             if not data: # Socket closed gracefully by peer
                 logging.warning("Receive error (internal): Connection closed by peer.")
                 self.handle_connection_loss()
                 return None
             decoded_data = data.decode('utf-8').strip()
             logging.debug(f"Received (internal): {decoded_data}")
             return decoded_data
         except socket.timeout:
             logging.error(f"Timeout receiving message (internal)")
             # Don't necessarily close socket on timeout, maybe robot is just slow
             # Consider if timeout should also trigger resume logic if it happens during drawing
             # For now, returning None might lead to connection loss handling higher up if expected msg isn't received.
             return None # Indicate timeout specifically? For now, None leads to resume check.
         except (socket.error, ConnectionResetError, BrokenPipeError) as e: # Added BrokenPipeError
             logging.error(f"Receive error (internal): {e}")
             self.handle_connection_loss() # Use centralized handler
             return None

    def handle_connection_loss(self):
        """Centralized handling of connection loss detection."""
        logging.warning("Connection lost detected.")
        was_connected = self.connected
        self.close_socket() # Close the broken socket and update flags
        # *** If connection lost DURING drawing, set resume flag ***
        # The resume flag is set higher up in the run_drawing_loop when errors occur
        # Here, we just ensure the socket is closed.
        # If we weren't drawing, we don't need to set resume_needed.
        # We might need to inform the user if they were connected but not drawing.
        if was_connected and not self.drawing_in_progress and not self.resume_needed:
             # Use 'after' to schedule GUI updates from the main thread
            self.window.after(0, lambda: messagebox.showinfo("Connection Lost", "Robot connection lost."))
            # Potentially navigate back to connection page if not already there
            # Check current page? For simplicity, assume user might need to reconnect manually.


    # --- Connection Handling ---
    def establish_connection(self):
        """Attempt connection (modified to use internal methods and handle resume)."""
        if hasattr(self, 'connect_button') and self.connect_button.winfo_exists(): self.connect_button.config(state=tk.DISABLED)
        if hasattr(self, 'reconnect_button') and self.reconnect_button.winfo_exists(): self.reconnect_button.config(state=tk.DISABLED)

        host, port = (SIMULATION_HOST, SIMULATION_PORT) if self.connection_var.get() == "simulation" else (REAL_ROBOT_HOST, REAL_ROBOT_PORT)

        def connection_attempt():
            try:
                self.close_socket() # Ensure clean start
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5)
                self.socket.connect((host, port))
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self.socket.settimeout(None) # Default to blocking for operations
                logging.info(f"Connected to {host}:{port}")
                self.connected = True
                # *** Call handle_connection_result via 'after' ***
                self.window.after(0, lambda: self.handle_connection_result(True))
            except (socket.error, socket.timeout, ConnectionRefusedError) as e:
                logging.error(f"Connection error: {e}")
                self.connected = False # Ensure flag is false before calling handler
                self.close_socket() # Clean up socket if connection failed
                # *** Call handle_connection_result via 'after' ***
                self.window.after(0, lambda: self.handle_connection_result(False))
        threading.Thread(target=connection_attempt, daemon=True).start()

    def handle_connection_result(self, connected):
        """Handle connection result and trigger resume if needed."""
        # Re-enable buttons safely
        if hasattr(self, 'connect_button') and self.connect_button.winfo_exists():
            self.connect_button.config(state=tk.NORMAL)
        if hasattr(self, 'reconnect_button') and self.reconnect_button.winfo_exists():
            self.reconnect_button.config(state=tk.NORMAL)

        if connected:
            self.connection_established = True
            # *** NEW: Check if resume is needed ***
            if self.resume_needed and self.resume_commands is not None:
                logging.info("Reconnection successful. Preparing to resume drawing.")
                # Move to final position BEFORE resuming
                self.move_to_final_before_resume() # This will start resume loop after move
            else:
                # Normal connection, go to drawing options
                self.drawing_options_page() # Go to drawing options
        else:
            if self.resume_needed:
                messagebox.showerror("Reconnection Failed", "Failed to reconnect. Cannot resume the previous drawing.")
                # Reset resume state as we can't continue
                self.resume_needed = False
                self.resume_commands = None
                self.resume_total_original_commands = 0
                self.resume_start_index_global = 0
                # Update last status to reflect the failed resume attempt
                self.last_drawing_status["status"] = "Resume Failed"
                self.last_drawing_status["error_message"] = "Could not reconnect to robot."
                # Go back to drawing options page after acknowledging the error
                self.drawing_options_page()
            else:
                messagebox.showerror("Connection Failed", "Failed to establish connection.")
            # Stay on connection page if it was a normal connection attempt that failed
    def move_to_final_before_resume(self):
        """Sends robot to FINAL_ROBOT_POSITION and then starts resume. Runs in thread."""
        def move_and_resume_thread():
            logging.info("Moving robot to FINAL_ROBOT_POSITION before resuming...")
            self.show_drawing_progress_page(self.resume_total_original_commands, self.resume_start_index_global, "Moving to resume position...")

            final_x, final_z, final_y = FINAL_ROBOT_POSITION
            command_str_final = f"{final_x:.3f},{final_z:.3f},{final_y:.3f}"
            move_ok = False
            if self.connected and self.socket:
                if self.send_message_internal(command_str_final):
                    response_r = self.receive_message_internal(timeout=5.0)
                    if response_r == "R":
                        response_d = self.receive_message_internal(timeout=5.0)
                        # if response_d == "D":
                        #     logging.info("Robot reached FINAL_ROBOT_POSITION.")
                        #     move_ok = True
                        # else: logging.error("Failed to get 'D' confirmation for pre-resume move.")
                    else: logging.error("Failed to get 'R' confirmation for pre-resume move.")
                else: logging.error("Failed to send pre-resume move command.")

            if move_ok:
                 # *** Start the drawing loop from the resume point ***
                 logging.info(f"Starting resume from command index {self.resume_start_index_global}")
                 self.drawing_in_progress = True # Set flag before starting thread
                 self.cancel_requested = False # Ensure cancel flag is reset
                 # We don't reset resume_needed here, it's reset on completion/cancel
                 # Use the stored remaining commands and start index
                 # NOTE: run_drawing_loop expects the FULL command list and the start_index
                 self.run_drawing_loop(self.resume_commands, self.resume_start_index_global)
                 # The run_drawing_loop itself now handles progress updates etc.
            else: # if move_ok is False
                error_msg = "Failed to move robot to safe resume position."
                logging.error(error_msg + " Cannot resume automatically, but allowing retry.")
                # *** NEW: Update status but keep resume state ***
                self.last_drawing_status["status"] = "Resume Failed (Pre-move)"
                self.last_drawing_status["error_message"] = error_msg
                # Keep previous command counts if available
                # Ensure resume_needed remains True, DO NOT reset resume variables here
                # *** End NEW ***
                self.window.after(0, lambda: messagebox.showwarning("Resume Warning", error_msg + "\nConnection might be unstable. You can try 'Reconnect & Resume' again."))
                # Reset drawing flag
                self.drawing_in_progress = False
                # *** NEW: Go back to connection page to allow retry ***
                self.window.after(1000, self.connection_setup_page)

        # Start the move and potential resume in a new thread
        threading.Thread(target=move_and_resume_thread, daemon=True).start()


    def close_socket(self):
        """Close socket cleanly and update flags."""
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except (socket.error, OSError): pass # Ignore errors if already closed
            finally:
                try: self.socket.close()
                except (socket.error, OSError): pass
                self.socket = None
                logging.info("Socket closed")
        # Always update flags when this is called
        self.connected = False
        self.connection_established = False
        # Do NOT reset drawing_in_progress or resume flags here,
        # they are managed by the drawing loop and connection loss handler

    def close_and_return_main(self):
         """Close connection and go to main page."""
         # If drawing was in progress, should we cancel it first?
         # For simplicity now, just close the socket. Active drawing will fail.
         self.close_socket()
         # Reset any pending resume state if user explicitly disconnects
         self.resume_needed = False
         self.resume_commands = None
         self.resume_total_original_commands = 0
         self.resume_start_index_global = 0
         self.main_page()

    # --- Utility Methods ---
    def clear_frame(self):
        """Clear all widgets from the main frame."""
        # Stop camera if running when clearing frame
        if self.camera_running:
            self.stop_camera_feed()
        # Destroy widgets
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        # Reset references to GUI elements that are destroyed
        self.camera_frame_label = None
        self.capture_button = None
        self.camera_back_button = None
        self.progress_bar = None
        self.status_label = None
        self.cancel_button = None
        self.connect_button = None
        self.reconnect_button = None
        self.preview_label = None


    @staticmethod
    def run_script(script_path: str) -> bool:
        """Run a Python script (kept for calibration)."""
        # (Implementation remains the same) 
        if not os.path.exists(script_path):
             logging.error(f"Script not found: {script_path}")
             return False
        try:
            logging.info(f"Running script: {script_path}")
            result = os.system(f'python "{script_path}"')
            if result != 0: logging.error(f"Script {script_path} failed with exit code {result}")
            return result == 0
        except Exception as e:
            logging.error(f"Error running script {script_path}: {e}")
            return False

    def on_window_close(self):
        """Handle window close event."""
        logging.info("Window close requested.")
        self.cancel_requested = True # Signal drawing thread to stop if running
        self.stop_camera_feed() # Ensure camera stops
        self.close_socket()     # Ensure socket closes
        # Give threads a moment to potentially react to cancel_requested or socket closure
        time.sleep(0.2)
        self.window.destroy()


# --- Main Execution ---
if __name__ == "__main__":
    # Create DATA_DIR if it doesn't exist 
    os.makedirs(DATA_DIR, exist_ok=True)

    app = RUNME_GUI()
    app.window.protocol("WM_DELETE_WINDOW", app.on_window_close)
    app.window.mainloop()