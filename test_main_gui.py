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
# Assuming the drawing functions are defined below or imported
# (image_to_contours, scale_point_to_a4, create_drawing_paths, calculate_distance)

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

# Drawing Specific Constants (from provided code)
FINAL_ROBOT_POSITION = (0,-350, 0) # Use X, Z, Y format from previous examples
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
PEN_UP_Z = -15
PEN_DOWN_Z = -7
MIN_CONTOUR_LENGTH_PX = 10

# Threshold options: (Label, threshold1, threshold2)
THRESHOLD_OPTIONS = [
    ("Detail", 25, 75),
    ("Balanced", 50, 150),
    ("Simple", 75, 200),
    ("Minimal", 100, 250)
]

# Rough estimate for time per command (adjust based on observation)
TIME_ESTIMATE_FACTOR = 0.3 # seconds per command (very rough guess)

# --- Drawing Helper Function ---
def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x, y)."""
    # Add check for None or invalid input if necessary
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
    """ Scales and transforms a single (x, y) pixel coordinate to centered A4 (mm). """
    x_pixel, y_pixel = point_xy
    center_x_pixel = image_width / 2
    center_y_pixel = image_height / 2
    x_centered_pixel = x_pixel - center_x_pixel
    y_centered_pixel = center_y_pixel - y_pixel # Invert y-axis
    x_mm = x_centered_pixel * scale_factor
    y_mm = y_centered_pixel * scale_factor
    return (x_mm, y_mm)

def create_drawing_paths(contours_xy, image_width, image_height, optimize_paths=True):
    """ Takes list of contours (pixel coordinates), scales them, creates drawing paths. """
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
        robot_commands.append((start_point[0], PEN_UP_Z, start_point[1]))
        robot_commands.append((start_point[0], PEN_DOWN_Z, start_point[1]))

        for i in range(len(contour) - 1):
            end_point = contour[i+1]
            # Avoid duplicate commands for single-point contours handled earlier
            if end_point != contour[i]:
                robot_commands.append((end_point[0], PEN_DOWN_Z, end_point[1]))

        final_point = contour[-1]
        robot_commands.append((final_point[0], PEN_UP_Z, final_point[1]))

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
        self.progress_bar = None
        self.status_label = None

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
        # ... (Radio buttons and connection frame as before) ...
        connection_frame = tk.Frame(self.main_frame)
        connection_frame.pack(pady=10)
        tk.Radiobutton(connection_frame, text=f"Simulation: {SIMULATION_HOST}:{SIMULATION_PORT}",
                       variable=self.connection_var, value="simulation").pack(anchor='w')
        tk.Radiobutton(connection_frame, text=f"Real Robot: {REAL_ROBOT_HOST}:{REAL_ROBOT_PORT}",
                       variable=self.connection_var, value="real").pack(anchor='w')

        self.connect_button = tk.Button(self.main_frame, text="Connect", command=self.establish_connection, width=20)
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
             threading.Thread(target=self.start_drawing_process, args=(self.selected_commands,), daemon=True).start()
             self.show_drawing_progress_page(len(self.selected_commands)) # Show progress UI
        else:
             messagebox.showwarning("Busy", "Drawing already in progress.")


    # --- Drawing Execution Workflow ---
    def show_drawing_progress_page(self, total_commands):
         """Displays the progress bar and status during drawing."""
         self.clear_frame()
         tk.Label(self.main_frame, text="Drawing in Progress...", font=("Arial", 16)).pack(pady=10)

         self.status_label = tk.Label(self.main_frame, text="Starting...")
         self.status_label.pack(pady=5)

         self.progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", length=300, mode="determinate", maximum=total_commands)
         self.progress_bar.pack(pady=10)

         # Add a cancel button? (More complex: requires signaling the thread)
         # tk.Button(self.main_frame, text="Cancel Drawing", command=self.cancel_drawing).pack(pady=5)


    def update_drawing_status(self, current_command_index, total_commands, message=""):
        """Callback to update progress bar and status label from drawing thread."""
        if self.progress_bar and self.status_label:
            self.progress_bar['value'] = current_command_index
            status_text = f"Sent {current_command_index} / {total_commands} commands"
            if message:
                 status_text += f" ({message})"
            self.status_label.config(text=status_text)
            # self.window.update_idletasks() # Force update if needed, but 'after' usually handles it


    def start_drawing_process(self, commands_to_send: List[Tuple]):
        """Sends drawing commands to the robot sequentially (RUNS IN THREAD)."""
        total_commands = len(commands_to_send)
        success = True
        last_sent_point = None # Track for potential redundant moves

        if not self.connected or not self.socket:
             logging.error("Drawing started but not connected.")
             self.window.after(0, lambda: messagebox.showerror("Error", "Connection lost before drawing started."))
             self.window.after(0, self.drawing_options_page) # Go back
             self.drawing_in_progress = False
             return

        try:
            for i, (x, z, y) in enumerate(commands_to_send):
                current_point = (x, z, y)

                # --- Robot Communication Protocol ---
                # Assumes: Send command -> Wait for 'R' -> Wait for 'D'
                command_str = f"{x:.2f},{z},{y:.2f}" # Format for robot

                # Check if robot is ready / connection ok before sending? (Optional PING)

                # 1. Send Command
                if not self.send_message_internal(command_str): # Use internal send without GUI popup
                    success = False
                    self.window.after(0, lambda i=i: self.update_drawing_status(i + 1, total_commands, "Send Failed"))
                    break

                # 2. Wait for Receipt 'R'
                response_r = self.receive_message_internal(timeout=10.0) # Use internal receive
                if response_r != "R":
                    logging.error(f"Robot did not confirm receipt (R), got '{response_r}'. Stopping.")
                    success = False
                    self.window.after(0, lambda i=i, r=response_r: self.update_drawing_status(i + 1, total_commands, f"Error: No 'R' (Got {r})"))
                    break

                # 3. Wait for Done 'D'
                response_d = self.receive_message_internal(timeout=60.0) # Longer timeout for move completion
                if response_d != "D":
                    logging.error(f"Robot did not confirm completion (D), got '{response_d}'. Stopping.")
                    success = False
                    self.window.after(0, lambda i=i, d=response_d: self.update_drawing_status(i + 1, total_commands, f"Error: No 'D' (Got {d})"))
                    break
                # --- End Protocol ---

                # Update GUI progress (use window.after for thread safety)
                self.window.after(0, lambda i=i: self.update_drawing_status(i + 1, total_commands))

                last_sent_point = current_point
                # time.sleep(0.01) # Small delay if needed? Usually not necessary

        except Exception as e:
            logging.error(f"Unexpected error during drawing process: {e}")
            success = False
            self.window.after(0, lambda: self.update_drawing_status(i if 'i' in locals() else 0, total_commands, f"Error: {e}"))
            # Try to close socket gracefully on error? Depends on state.

        finally:
            # --- Send Final Position Command ---
            if success: # Only go home if drawing finished ok
                 self.window.after(0, lambda: self.update_drawing_status(total_commands, total_commands, "Drawing Done. Moving to final position..."))
                 final_x, final_z, final_y = FINAL_ROBOT_POSITION
                 command_str_final = f"{final_x:.3f},{final_z:.3f},{final_y:.3f}" # Format final command
                 logging.info(f"Sending final position command: {command_str_final}")

                 if self.send_message_internal(command_str_final):
                      response_r_final = self.receive_message_internal(timeout=10.0)
                      if response_r_final == "R":
                           response_d_final = self.receive_message_internal(timeout=60.0)
                           if response_d_final == "D":
                                logging.info("Robot reached final position.")
                                self.window.after(0, lambda: self.status_label.config(text="Drawing Complete. Robot at final position."))
                           else:
                                logging.error(f"Robot didn't confirm final move completion (D), got '{response_d_final}'")
                                self.window.after(0, lambda: self.status_label.config(text="Drawing Complete. Final move confirmation failed."))
                      else:
                           logging.error(f"Robot didn't confirm final move receipt (R), got '{response_r_final}'")
                           self.window.after(0, lambda: self.status_label.config(text="Drawing Complete. Final move send failed (No 'R')."))
                 else:
                      logging.error("Failed to send final position command.")
                      self.window.after(0, lambda: self.status_label.config(text="Drawing Complete. Failed to send final move command."))
            else:
                 # Update status if drawing failed
                 self.window.after(0, lambda: self.status_label.config(text="Drawing stopped due to error."))


            # Drawing finished or failed, reset state and return to options page
            self.drawing_in_progress = False
            self.selected_commands = None
            # Go back to the drawing options page after a short delay
            self.window.after(2000, self.drawing_options_page) # Wait 2s before going back


    # --- Internal Socket Methods (without GUI popups) ---
    def send_message_internal(self, message: str) -> bool:
        """ Sends message without triggering GUI popups on error. Returns success status. """
        if not self.connected or not self.socket: return False
        try:
            self.socket.sendall(message.encode('utf-8'))
            logging.debug(f"Sent (internal): {message}")
            return True
        except (socket.error, ConnectionResetError, socket.timeout) as e:
            logging.error(f"Send error (internal): {e}")
            self.close_socket() # Close broken socket
            return False

    def receive_message_internal(self, timeout=10.0) -> Optional[str]:
         """ Receives message without triggering GUI popups on error. Returns message or None. """
         if not self.connected or not self.socket: return None
         try:
             self.socket.settimeout(timeout)
             data = self.socket.recv(1024)
             self.socket.settimeout(None) # Reset timeout
             decoded_data = data.decode('utf-8').strip()
             logging.debug(f"Received (internal): {decoded_data}")
             return decoded_data
         except socket.timeout:
             logging.error(f"Timeout receiving message (internal)")
             # Don't necessarily close socket on timeout, maybe robot is just slow
             return None
         except (socket.error, ConnectionResetError) as e:
             logging.error(f"Receive error (internal): {e}")
             self.close_socket() # Close broken socket
             return None


    # --- Connection Handling ---
    def establish_connection(self):
        """Attempt connection (modified to use internal methods)."""
        if hasattr(self, 'connect_button'): self.connect_button.config(state=tk.DISABLED)
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
                self.window.after(0, lambda: self.handle_connection_result(True))
            except (socket.error, socket.timeout, ConnectionRefusedError) as e:
                logging.error(f"Connection error: {e}")
                self.connected = False
                self.close_socket()
                self.window.after(0, lambda: self.handle_connection_result(False))
        threading.Thread(target=connection_attempt, daemon=True).start()

    def handle_connection_result(self, connected):
        """Handle connection result."""
        if hasattr(self, 'connect_button') and self.connect_button.winfo_exists():
            self.connect_button.config(state=tk.NORMAL)
        if connected:
            self.connection_established = True
            self.drawing_options_page() # Go to drawing options
        else:
            messagebox.showerror("Connection Failed", "Failed to establish connection.")

    def close_socket(self):
        """Close socket cleanly."""
        # (Implementation remains the same as previous version)
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except (socket.error, OSError): pass # Ignore errors if already closed
            finally:
                 try: self.socket.close()
                 except (socket.error, OSError): pass
                 self.socket = None
                 self.connected = False
                 self.connection_established = False
                 logging.info("Connection closed")
        else:
             self.connected = False
             self.connection_established = False

    def close_and_return_main(self):
         """Close connection and go to main page."""
         self.close_socket()
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
        # Reset references
        self.camera_frame_label = None
        self.capture_button = None
        self.camera_back_button = None
        self.progress_bar = None
        self.status_label = None


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
        self.stop_camera_feed() # Ensure camera stops
        self.close_socket()     # Ensure socket closes
        self.window.destroy()

    # --- Calibration Methods (Keep if needed) ---
    def calibration_page(self):
         # ... (Keep calibration page logic if required) ...
         messagebox.showinfo("Info", "Calibration feature not fully integrated with new workflow yet.")
         pass

    def capture_checkerboard_image(self):
         # ... (Keep logic if required) ...
         pass

    def run_calibration_script(self):
         # ... (Keep logic if required) ...
         pass

    def test_camera(self): # Might conflict with drawing capture - review usage
         # ... (Keep logic if required, ensure it doesn't interfere) ...
         pass


# --- Main Execution ---
if __name__ == "__main__":
    # Create DATA_DIR if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    app = RUNME_GUI()
    app.window.protocol("WM_DELETE_WINDOW", app.on_window_close)
    app.window.mainloop()