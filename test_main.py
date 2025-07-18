import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import os
import threading
import time
import logging
import socket
from typing import List, Tuple, Optional
import cv2
import numpy as np
import math
from PIL import Image, ImageTk
import ast

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants (Consolidated) ---
SCRIPT_DIR = os.getenv("SCRIPT_DIR", ".")
DATA_DIR = os.getenv("DATA_DIR", ".")

TMP_CAPTURE_PATH = os.path.join(DATA_DIR, "temp_capture.png")
TMP_EDGE_OUTPUT_PATH = os.path.join(DATA_DIR, "temp_edges_{}.png")

REAL_ROBOT_HOST = '192.168.125.1'
REAL_ROBOT_PORT = 1025
SIMULATION_HOST = '127.0.0.1'
SIMULATION_PORT = 55000

# Drawing Specific Constants
FINAL_ROBOT_POSITION = (0, -50, 0) # Use X, Z, Y format (X, Depth, Y) - NOTE: Z is depth here
A4_WIDTH_MM = 170  # Drawing area width
A4_HEIGHT_MM = 207 # Drawing area height
DEFAULT_PEN_DOWN_Z = -10   # Default pen down position (depth)

MIN_CONTOUR_LENGTH_PX = 10 # Minimum contour length in pixels to consider

# Threshold options for Canny edge detection
THRESHOLD_OPTIONS = [
    ("Option {}".format(i), i*10, i*20) for i in range(1, 8)
]

# Time estimation factor (seconds per command)
TIME_ESTIMATE_FACTOR = 0.02

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

        # Camera related variables
        self.cap = None
        self.camera_running = False
        self.camera_frame_label = None
        self.capture_button = None
        self.camera_back_button = None

        # Drawing process related variables
        self.current_image_path = None
        self.threshold_options_data = {}
        self.edge_preview_paths = {}
        self.selected_commands = None
        self.drawing_in_progress = False
        self.cancel_requested = False
        self.progress_bar = None
        self.status_label = None
        self.cancel_button = None
        self.reconnect_button = None

        # Pen position and control variables
        self.pen_down_z_var = tk.StringVar(value=str(DEFAULT_PEN_DOWN_Z))
        self.safe_center_z_var = tk.StringVar(value=str(-50.0))
        self.pause_event = threading.Event()
        self.pause_resume_button = None

        # ETA Countdown variables
        self.eta_update_id = None
        self.drawing_start_time = 0
        self.total_paused_time = 0
        self.pause_start_time = 0
        self.progress_text_var = tk.StringVar()

        # Status tracking for previous drawing attempts
        self.last_drawing_status = {
            "total_commands": 0,
            "completed_commands": 0,
            "status": "None",
            "error_message": ""
        }
        
        # Resume-related variables
        self.resume_needed = False
        self.resume_commands = None
        self.resume_start_index_global = 0

        # Start the application
        self.main_page()

    # --- Drawing Logic Methods (Refactored) ---

    def load_and_create_signature_commands(self):
        """
        Loads signature points from tmp_signaturepoint.py, injects the dynamic Z value,
        parses them safely, and formats them into executable commands.
        """
        try:
            pen_down_z = float(self.pen_down_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Pen Down Z-coordinate must be a valid number.")
            return None

        try:
            with open('tmp_signaturepoint.py', 'r') as f:
                content = f.read()
            
            # Replace the placeholder string with the actual dynamic value
            content_with_z = content.replace('PEN_DOWN_Z', str(pen_down_z))
            
            # Extract the tuple part of the string after the '='
            points_str = content_with_z.split('=', 1)[1].strip()
            # Remove a trailing comment if it exists (e.g., '# type: ignore')
            if '#' in points_str:
                points_str = points_str.split('#', 1)[0].strip()

            # Safely evaluate the string to get the tuple of points
            raw_points = ast.literal_eval(points_str)
            logging.info("Successfully loaded and parsed signature points from file.")

        except (FileNotFoundError, IndexError, SyntaxError, ValueError) as e:
            logging.warning(f"Could not load or parse tmp_signaturepoint.py: {e}. Using a default signature.")
            # Define a fallback signature using the dynamic pen_down_z
            raw_points = (
                (50, pen_down_z, 50), (50, pen_down_z, -50), (-50, pen_down_z, -50),
                (-50, pen_down_z, 50), (50, pen_down_z, 50)
            )
        
        # Now format these points with pen up/down commands
        return self._format_signature_commands(raw_points, pen_down_z)

    def _format_signature_commands(self, processed_points, pen_down_z):
        """
        Takes a list of points with correct Z values and adds pen-up/down commands.
        """
        # Calculate a safe pen-up position
        pen_up_z = pen_down_z / 5 if pen_down_z > 0 else pen_down_z * 1.25

        commands = []
        if not processed_points:
            return commands

        # 1. Move to the start of the signature with the pen up
        start_x, _, start_y = processed_points[0]
        commands.append((start_x, pen_up_z, start_y))

        # 2. Add all processed signature points as commands
        # The points already have the correct Z values from the loading step
        for point in processed_points:
            commands.append(point)

        # 3. Lift the pen after the last point
        last_x, _, last_y = processed_points[-1]
        commands.append((last_x, pen_up_z, last_y))

        return commands

    def create_drawing_paths(self, contours_xy, image_width, image_height, optimize_paths=True):
        """
        Takes a list of contours (pixel coordinates), scales them to the drawing area,
        optimizes the drawing order, and generates the final robot commands.
        """
        try:
            pen_down_z = float(self.pen_down_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Pen Down Z-coordinate must be a valid number.")
            return None # Return None on error

        # Calculate a safe pen-up position (higher, i.e., less negative)
        pen_up_z = pen_down_z / 10 if pen_down_z > 0 else pen_down_z * 1.5
        
        if not contours_xy or image_width <= 0 or image_height <= 0:
            return []

        # Calculate scale factor to fit the image to the A4 drawing area
        scale_x = A4_WIDTH_MM / image_width
        scale_y = A4_HEIGHT_MM / image_height
        scale_factor = min(scale_x, scale_y)

        # Scale all contour points from pixel coordinates to robot-friendly mm coordinates
        scaled_contours = []
        for contour in contours_xy:
            if not contour: continue
            scaled_contour = [self.scale_point_to_a4(p, image_width, image_height, scale_factor) for p in contour]
            if len(scaled_contour) >= 1:
                scaled_contours.append(scaled_contour)

        if not scaled_contours:
            return []

        # Optimize the drawing path to minimize travel distance
        if optimize_paths:
            ordered_contours = self.optimize_contour_order(scaled_contours)
        else:
            ordered_contours = scaled_contours

        # Generate the final list of robot commands (X, Z, Y)
        robot_commands = []
        for contour in ordered_contours:
            if not contour: continue
            
            # Handle single-point contours (dots)
            if len(contour) == 1:
                point = contour[0]
                robot_commands.append((point[0], pen_up_z, point[1]))   # Move to location
                robot_commands.append((point[0], pen_down_z, point[1])) # Pen down
                robot_commands.append((point[0], pen_up_z, point[1]))   # Pen up
                continue

            # Handle multi-point contours (lines)
            start_point = contour[0]
            robot_commands.append((start_point[0], pen_up_z, start_point[1]))   # Move to start of line
            robot_commands.append((start_point[0], pen_down_z, start_point[1])) # Pen down

            # Draw along the contour
            for point in contour[1:]:
                robot_commands.append((point[0], pen_down_z, point[1]))

            # Lift pen at the end of the contour
            final_point = contour[-1]
            robot_commands.append((final_point[0], pen_up_z, final_point[1]))

        return robot_commands
        
    def optimize_contour_order(self, contours: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        """
        Sorts contours to minimize travel distance between them using a nearest-neighbor approach.
        """
        if not contours:
            return []

        ordered_contours = []
        remaining_contours = list(contours)
        
        # Start with the first contour
        current_contour = remaining_contours.pop(0)
        ordered_contours.append(current_contour)
        last_point = current_contour[-1]

        while remaining_contours:
            best_dist = float('inf')
            best_idx = -1
            best_reversed = False

            # Find the closest next contour (or the reversed version of it)
            for i, contour in enumerate(remaining_contours):
                dist_start = self.calculate_distance(last_point, contour[0])
                dist_end = self.calculate_distance(last_point, contour[-1])

                if dist_start < best_dist:
                    best_dist, best_idx, best_reversed = dist_start, i, False
                if dist_end < best_dist:
                    best_dist, best_idx, best_reversed = dist_end, i, True

            if best_idx != -1:
                next_contour = remaining_contours.pop(best_idx)
                if best_reversed:
                    next_contour.reverse()
                ordered_contours.append(next_contour)
                last_point = next_contour[-1]
            else:
                logging.warning("Path optimization loop finished unexpectedly.")
                break # Safety break

        return ordered_contours

    def image_to_contours_internal(self, image_path_or_array, threshold1, threshold2, save_edge_path=None):
        """
        Convert image to contours using Canny edge detection.
        Accepts a file path or a pre-loaded cv2 image array.
        """
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_path_or_array, np.ndarray):
            image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY) if len(image_path_or_array.shape) == 3 else image_path_or_array
        else:
            logging.error("Invalid input type for image_to_contours_internal")
            return None, 0, 0

        if image is None:
            logging.error("Could not read or process image input.")
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

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.arcLength(c, closed=False) > MIN_CONTOUR_LENGTH_PX]

        contours_xy = []
        for contour in filtered_contours:
            points = contour.squeeze().tolist()
            if not isinstance(points, list) or not points: continue
            if isinstance(points[0], int): points = [points]
            contours_xy.append([(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2])

        return [c for c in contours_xy if c], image_width, image_height

    @staticmethod
    def scale_point_to_a4(point_xy, image_width, image_height, scale_factor):
        """ Scales and transforms a single (x, y) pixel coordinate to a centered robot coordinate (mm)."""
        x_pixel, y_pixel = point_xy
        x_centered_pixel = x_pixel - (image_width / 2)
        y_centered_pixel = (image_height / 2) - y_pixel  # Invert y-axis for standard Cartesian coordinates
        x_mm = x_centered_pixel * scale_factor
        y_mm = y_centered_pixel * scale_factor
        return (x_mm, y_mm)

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculates Euclidean distance between two points (x, y)."""
        if p1 is None or p2 is None: return float('inf')
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # --- Page Navigation ---
    def main_page(self):
        """Main application page."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Robotics Drawing System", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.main_frame, text="Setup Connection & Draw",
                  command=self.connection_setup_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Exit",
                  command=self.on_window_close, width=30).pack(pady=5)

    def connection_setup_page(self):
        """Page for setting up robot connection."""
        self.clear_frame()
        tk.Label(self.main_frame, text="Robot Connection Setup", font=("Arial", 16)).pack(pady=10)

        connection_frame = tk.Frame(self.main_frame)
        connection_frame.pack(pady=10)
        tk.Radiobutton(connection_frame, text=f"Simulation: {SIMULATION_HOST}:{SIMULATION_PORT}",
                       variable=self.connection_var, value="simulation").pack(anchor='w')
        tk.Radiobutton(connection_frame, text=f"Real Robot: {REAL_ROBOT_HOST}:{REAL_ROBOT_PORT}",
                       variable=self.connection_var, value="real").pack(anchor='w')

        self.connect_button = tk.Button(self.main_frame, text="Connect", command=self.establish_connection, width=20)
        self.reconnect_button = tk.Button(self.main_frame, text="Reconnect & Resume", command=self.establish_connection, width=20)

        if self.resume_needed:
            self.reconnect_button.pack(pady=5)
            tk.Label(self.main_frame, text="Connection lost during last drawing. Reconnect to resume.", fg="orange").pack()
        else:
            self.connect_button.pack(pady=5)

        tk.Button(self.main_frame, text="Back", command=self.main_page, width=20).pack(pady=5)

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
        
        # Display status of the last drawing attempt
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

        # Controls for testing and calibration
        controls_frame = tk.Frame(self.main_frame, pady=5, relief=tk.GROOVE, borderwidth=2)
        controls_frame.pack(pady=10, padx=10, fill='x')
        
        tk.Label(controls_frame, text="Testing & Calibration Controls", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, pady=5)

        tk.Label(controls_frame, text="Pen Down Z (for drawing):").grid(row=1, column=0, sticky='w', padx=5)
        tk.Entry(controls_frame, textvariable=self.pen_down_z_var, width=10).grid(row=1, column=1, padx=5)
        self.send_z_button = tk.Button(controls_frame, text="Test at (0, 0, Z)", command=self.send_to_test_z_action)
        self.send_z_button.grid(row=1, column=2, padx=10)

        tk.Label(controls_frame, text="Safe Center Z:").grid(row=2, column=0, sticky='w', padx=5)
        tk.Entry(controls_frame, textvariable=self.safe_center_z_var, width=10).grid(row=2, column=1, padx=5)
        self.safe_center_button = tk.Button(controls_frame, text="Go to Safe Center", command=self.send_to_safe_center_action)
        self.safe_center_button.grid(row=2, column=2, padx=10)

        self.test_workspace_button = tk.Button(controls_frame, text="Test Workspace Area", command=self.test_workspace_action)
        self.test_workspace_button.grid(row=3, column=0, columnspan=3, pady=5)

        # Main action buttons
        tk.Button(self.main_frame, text="Capture Image to Draw",
                  command=self.capture_image_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Input Image to Draw",
                  command=self.input_image_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Disconnect",
                  command=self.close_and_return_main, width=30).pack(pady=5)

    def send_to_test_z_action(self):
        """Button action to test the pen_down_z value."""
        try:
            test_z = float(self.pen_down_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Pen Down Z-coordinate must be a valid number.")
            return

        if hasattr(self, 'send_z_button') and self.send_z_button.winfo_exists():
            self.send_z_button.config(state=tk.DISABLED)
        threading.Thread(target=self._send_command_sequence_thread, args=([(0.0, test_z, 0.0)], self.send_z_button), daemon=True).start()

    def send_to_safe_center_action(self):
        """Button action for the safe center, with an adjustable value."""
        try:
            safe_z = float(self.safe_center_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Safe Center Z-coordinate must be a valid number.")
            return
        
        if hasattr(self, 'safe_center_button') and self.safe_center_button.winfo_exists():
            self.safe_center_button.config(state=tk.DISABLED)
        
        logging.info(f"Sending robot to safe center (0, {safe_z}, 0)")
        threading.Thread(target=self._send_command_sequence_thread, args=([(0, safe_z, 0)], self.safe_center_button), daemon=True).start()

    def test_workspace_action(self):
        """Sends the robot on a path to outline the workspace corners."""
        try:
            test_z = float(self.pen_down_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Z-coordinate for testing must be a valid number.")
            return
        
        pen_up_z =  test_z / 10 if test_z > 0 else test_z * 1.5

        if hasattr(self, 'test_workspace_button') and self.test_workspace_button.winfo_exists():
            self.test_workspace_button.config(state=tk.DISABLED)
        
        # Define the workspace path (X, Z, Y)
        w = A4_WIDTH_MM / 3
        h = A4_HEIGHT_MM / 3
        workspace_path = [
            (w, pen_up_z, h),   # Top-right (pen up)
            (w, test_z, h),     # Top-right (pen down)
            (w, test_z, -h),    # Bottom-right
            (-w, test_z, -h),   # Bottom-left
            (-w, test_z, h),    # Top-left
            (w, test_z, h),     # Back to Top-right
            (0, pen_up_z, 0)    # Return to center (pen up)
        ]
        
        logging.info("Starting workspace test...")
        threading.Thread(target=self._send_command_sequence_thread, args=(workspace_path, self.test_workspace_button), daemon=True).start()

    def _send_command_sequence_thread(self, commands: List[Tuple], button_to_re_enable: tk.Button):
        """Thread worker to send a sequence of commands, one by one."""
        original_text = button_to_re_enable.cget("text")
        self.window.after(0, lambda: button_to_re_enable.config(text="Moving..."))

        for i, (x, z, y) in enumerate(commands):
            if self.cancel_requested:
                logging.info("Test sequence cancelled.")
                break
            
            command_str = f"{x:.2f},{z:.2f},{y:.2f}"
            logging.info(f"Sending command {i+1}/{len(commands)}: {command_str}")
            
            if self.send_message_internal(command_str):
                response_r = self.receive_message_internal(timeout=10.0)
                if response_r != "R":
                    error_msg = f"Robot did not confirm receipt (R) for command {i+1}. Got: '{response_r}'"
                    logging.error(error_msg)
                    self.window.after(0, lambda: messagebox.showerror("Test Failed", error_msg))
                    break
            else:
                self.window.after(0, lambda: messagebox.showerror("Connection Error", "Failed to send test command. Connection may be lost."))
                break
        
        if button_to_re_enable and button_to_re_enable.winfo_exists():
            self.window.after(0, lambda: button_to_re_enable.config(state=tk.NORMAL, text=original_text))
        logging.info(f"Sequence '{original_text}' finished.")

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

        self.window.bind('s', lambda event: self.capture_action())
        self.window.bind('S', lambda event: self.capture_action())

        self.start_camera_feed()

    def start_camera_feed(self):
        """Starts displaying the camera feed."""
        if self.camera_running: return

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera.")
                self.stop_camera_and_go_back()
                return
            self.camera_running = True
            self._update_camera_frame()
        except Exception as e:
             messagebox.showerror("Camera Error", f"Error initializing camera: {e}")
             self.stop_camera_and_go_back()

    def _update_camera_frame(self):
        """Internal method to continuously update the camera feed label."""
        if not self.camera_running or not self.cap: return

        ret, frame = self.cap.read()
        if ret:
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)

            if self.camera_frame_label and self.camera_frame_label.winfo_exists():
                self.camera_frame_label.imgtk = imgtk
                self.camera_frame_label.configure(image=imgtk)
        else:
            logging.warning("Failed to grab frame from camera.")

        if self.camera_running:
            self.window.after(30, self._update_camera_frame)

    def stop_camera_feed(self):
         """Stops the camera feed and releases resources."""
         self.camera_running = False
         time.sleep(0.1)
         if self.cap:
              self.cap.release()
              self.cap = None

    def stop_camera_and_go_back(self):
        """Stops camera and returns to drawing options page."""
        self.stop_camera_feed()
        self.window.unbind('s')
        self.window.unbind('S')
        self.drawing_options_page()

    def capture_action(self):
        """Captures the current frame and processes it."""
        if not self.camera_running or not self.cap:
            messagebox.showwarning("Capture Error", "Camera not running.")
            return

        ret, frame = self.cap.read()
        self.stop_camera_feed()
        self.window.unbind('s')
        self.window.unbind('S')

        if ret:
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                cv2.imwrite(TMP_CAPTURE_PATH, frame)
                logging.info(f"Image captured and saved to {TMP_CAPTURE_PATH}")
                self.current_image_path = TMP_CAPTURE_PATH
                self.show_threshold_options(self.current_image_path)
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save captured image: {e}")
                self.drawing_options_page()
        else:
            messagebox.showerror("Capture Error", "Failed to capture frame from camera.")
            self.drawing_options_page()

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

        self.threshold_options_data = {}
        self.selected_threshold_option = tk.StringVar(value=None)
        self.preview_label = tk.Label(self.main_frame)
        self.preview_label.pack(pady=5)

        options_frame = tk.Frame(self.main_frame)
        options_frame.pack(pady=5)

        loading_label = tk.Label(options_frame, text="Processing options...")
        loading_label.pack()
        self.window.update()

        threading.Thread(target=self._process_threshold_options_thread, args=(image_path, options_frame, loading_label), daemon=True).start()

    def _process_threshold_options_thread(self, image_path, options_frame, loading_label):
        """Background thread to generate commands for each threshold option."""
        results = {}
        preview_paths = {}

        for i, (label, t1, t2) in enumerate(THRESHOLD_OPTIONS):
            logging.info(f"Processing option: {label} (t1={t1}, t2={t2})")
            preview_path = TMP_EDGE_OUTPUT_PATH.format(i)
            contours_xy, w, h = self.image_to_contours_internal(image_path, t1, t2, save_edge_path=preview_path)

            if contours_xy is None or w == 0 or h == 0:
                 logging.warning(f"Failed to process contours for option {label}")
                 results[label] = None
                 preview_paths[label] = None
                 continue

            commands = self.create_drawing_paths(contours_xy, w, h, optimize_paths=True)
            if commands:
                num_commands = len(commands)
                est_time_sec = num_commands * TIME_ESTIMATE_FACTOR
                est_time_min = est_time_sec / 60
                results[label] = {
                    "commands": commands, 
                    "count": num_commands,
                    "time_str": f"{est_time_min:.1f} min"
                }
                preview_paths[label] = preview_path if os.path.exists(preview_path) else None
            else:
                 results[label] = None
                 preview_paths[label] = None
                 logging.warning(f"No commands generated for option {label}")

        self.window.after(0, lambda: self._display_threshold_options(options_frame, loading_label, results, preview_paths))

    def _display_threshold_options(self, options_frame, loading_label, results, preview_paths):
         """Updates the GUI with the processed threshold options."""
         loading_label.destroy()

         self.threshold_options_data = results
         self.edge_preview_paths = preview_paths

         default_selected = False
         for label, t1, t2 in THRESHOLD_OPTIONS:
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
                    command=lambda l=label: self.show_edge_preview(l)
                 )
                 rb.pack(anchor='w')
                 if not default_selected:
                      self.selected_threshold_option.set(label)
                      self.show_edge_preview(label)
                      default_selected = True
             else:
                 tk.Label(options_frame, text=f"{label} (t1={t1}, t2={t2}) - No drawing generated", fg="gray").pack(anchor='w')

         button_frame = tk.Frame(self.main_frame)
         button_frame.pack(pady=10)
         tk.Button(button_frame, text="Confirm and Draw", command=self.confirm_and_start_drawing, width=20).pack(side=tk.LEFT, padx=5)
         tk.Button(button_frame, text="Save Points to File", command=self.save_points_to_file, width=20).pack(side=tk.LEFT, padx=5)
         tk.Button(button_frame, text="Back", command=self.drawing_options_page, width=20).pack(side=tk.LEFT, padx=5)

    def save_points_to_file(self):
        """Saves the generated drawing commands for the selected option to a text file."""
        selected_label = self.selected_threshold_option.get()
        if not selected_label:
            messagebox.showwarning("Selection Needed", "Please select a drawing style option first.")
            return

        option_data = self.threshold_options_data.get(selected_label)
        if not option_data or not option_data.get("commands"):
            messagebox.showerror("Error", "Selected option has no drawing commands to save.")
            return

        commands = option_data["commands"]
        
        filepath = filedialog.asksaveasfilename(
            title="Save Drawing Points",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialfile="drawing_points.txt"
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w') as f:
                f.write("X, Z, Y\n") # Header
                for x, z, y in commands:
                    f.write(f"{x:.3f},{z:.3f},{y:.3f}\n")
            
            messagebox.showinfo("Success", f"Drawing points successfully saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save the file.\nError: {e}")

    def show_edge_preview(self, option_label):
         """Displays the edge preview image for the selected option."""
         preview_path = self.edge_preview_paths.get(option_label)
         if preview_path and os.path.exists(preview_path):
              try:
                   img = Image.open(preview_path)
                   img.thumbnail((300, 300))
                   imgtk = ImageTk.PhotoImage(image=img)
                   self.preview_label.imgtk = imgtk
                   self.preview_label.configure(image=imgtk)
              except Exception as e:
                   logging.error(f"Error loading preview image {preview_path}: {e}")
                   self.preview_label.configure(image=None, text="Preview error")
         else:
              self.preview_label.configure(image=None, text="No Preview")

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

        if not self.drawing_in_progress:
             self.drawing_in_progress = True
             self.cancel_requested = False
             self.resume_needed = False
             self.pause_event.set() # Ensure event is set (not paused) at start
             
             self.drawing_start_time = time.time()
             self.total_paused_time = 0
             self.pause_start_time = 0

             # Generate signature commands using the new dynamic loading method
             signature_commands = self.load_and_create_signature_commands()
             if signature_commands is None: # An error occurred (e.g., invalid Z value)
                 self.drawing_in_progress = False # Reset flag
                 return

             full_command_list = self.selected_commands + signature_commands
             logging.info(f"Starting drawing with {len(self.selected_commands)} image commands and {len(signature_commands)} signature commands.")

             threading.Thread(target=self.run_drawing_loop, args=(full_command_list,), daemon=True).start()
             self.show_drawing_progress_page(len(full_command_list))
        else:
            messagebox.showwarning("Busy", "Drawing already in progress.")

    # --- Drawing Execution Workflow ---
    def show_drawing_progress_page(self, total_commands, current_progress=0, status_message="Starting..."):
         """Displays the progress bar and status during drawing."""
         self.clear_frame()
         tk.Label(self.main_frame, text="Drawing in Progress...", font=("Arial", 16)).pack(pady=10)

         self.status_label = tk.Label(self.main_frame, textvariable=self.progress_text_var)
         self.status_label.pack(pady=5)

         self.progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", length=300, mode="determinate", maximum=total_commands, value=current_progress)
         self.progress_bar.pack(pady=10)

         controls_frame = tk.Frame(self.main_frame)
         controls_frame.pack(pady=5)

         self.pause_resume_button = tk.Button(controls_frame, text="Pause", command=self.toggle_pause_resume, width=15)
         self.pause_resume_button.pack(side=tk.LEFT, padx=5)
         
         self.cancel_button = tk.Button(controls_frame, text="Cancel Drawing", command=self.request_cancel_drawing, width=15)
         self.cancel_button.pack(side=tk.LEFT, padx=5)
         
         self.update_drawing_status(current_progress, total_commands)
         self._update_eta_countdown()

    def _update_eta_countdown(self):
        """Periodically updates the ETA label with a dynamic estimate."""
        if not self.drawing_in_progress:
            return

        completed_cmds = self.progress_bar['value']
        total_cmds = self.progress_bar['maximum']
        
        remaining_time = 0
        
        if not self.pause_event.is_set(): # If paused, just show Paused status
            self.progress_text_var.set(f"Sent {completed_cmds} / {total_cmds} commands | PAUSED")
        elif completed_cmds > 5: # Dynamic ETA after a few commands
            active_drawing_time = (time.time() - self.drawing_start_time) - self.total_paused_time
            if active_drawing_time > 0:
                avg_time_per_cmd = active_drawing_time / completed_cmds
                remaining_cmds = total_cmds - completed_cmds
                remaining_time = remaining_cmds * avg_time_per_cmd
        else: # Static ETA at the beginning
            elapsed_time = (time.time() - self.drawing_start_time) - self.total_paused_time
            initial_total_time = total_cmds * TIME_ESTIMATE_FACTOR
            remaining_time = max(0, initial_total_time - elapsed_time)

        mins, secs = divmod(int(remaining_time), 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        if self.pause_event.is_set(): # Only update ETA if not paused
            self.progress_text_var.set(f"Sent {completed_cmds} / {total_cmds} commands | ETA: {time_str}")

        self.eta_update_id = self.window.after(1000, self._update_eta_countdown)

    def toggle_pause_resume(self):
        """Toggles the pause/resume state of the drawing loop."""
        if self.pause_event.is_set():
            # --- PAUSING ---
            self.pause_event.clear()
            logging.info("Drawing paused by user.")
            if self.pause_resume_button and self.pause_resume_button.winfo_exists():
                self.pause_resume_button.config(text="Resume")
            self.pause_start_time = time.time()
        else:
            # --- RESUMING ---
            if self.pause_start_time > 0:
                paused_duration = time.time() - self.pause_start_time
                self.total_paused_time += paused_duration
                self.pause_start_time = 0
            self.pause_event.set()
            logging.info("Drawing resumed by user.")
            if self.pause_resume_button and self.pause_resume_button.winfo_exists():
                self.pause_resume_button.config(text="Pause")

    def update_drawing_status(self, current_command_index, total_commands, message=""):
        """Callback to update progress bar and status label from drawing thread."""
        if self.progress_bar and self.progress_bar.winfo_exists():
            self.progress_bar['value'] = current_command_index
        if message:
            self.progress_text_var.set(f"Sent {current_command_index} / {total_commands} commands | {message}")

    def request_cancel_drawing(self):
        """Sets the cancellation flag when the Cancel button is pressed."""
        if self.drawing_in_progress:
            logging.info("Cancel requested by user.")
            self.cancel_requested = True
            self.pause_event.set() # Unblock the loop if it was paused
            if self.cancel_button and self.cancel_button.winfo_exists():
                self.cancel_button.config(text="Cancelling...", state=tk.DISABLED)
            if self.pause_resume_button and self.pause_resume_button.winfo_exists():
                self.pause_resume_button.config(state=tk.DISABLED)
            self.progress_text_var.set("Cancellation requested...")

    def _send_final_position_and_cleanup(self, success_message, failure_message):
        """Sends the robot to the final position and cleans up state. Runs in drawing thread."""
        self.drawing_in_progress = False # Stop ETA loop
        logging.info("Attempting to move robot to final position.")
        final_x, final_z, final_y = FINAL_ROBOT_POSITION
        command_str_final = f"{final_x:.3f},{final_z:.3f},{final_y:.3f}"

        move_ok = False
        if self.connected and self.socket:
            if self.send_message_internal(command_str_final):
                response_r_final = self.receive_message_internal(timeout=20.0)
                if response_r_final == "R":
                    logging.info("Robot received final move command.")
                    move_ok = True
                else:
                    logging.error(f"Robot didn't confirm final move receipt (R), got '{response_r_final}'") 
            else:
                logging.error("Failed to send final position command.") 

        final_status = f"{success_message} Final move command sent." if move_ok else f"{failure_message} Failed to send final move command."

        self.last_drawing_status["status"] = success_message
        self.last_drawing_status["error_message"] = "" if move_ok else "Failed to send final move command."

        self.window.after(0, lambda fs=final_status: self.update_final_status(fs))

        self.selected_commands = None
        self.cancel_requested = False
        if not self.resume_needed:
            self.resume_commands = None
            self.resume_start_index_global = 0

        self.window.after(2000, self.drawing_options_page)

    def update_final_status(self, message):
        """Updates the status label safely from the main thread."""
        if self.eta_update_id:
            self.window.after_cancel(self.eta_update_id)
            self.eta_update_id = None
        if self.status_label and self.status_label.winfo_exists():
            self.progress_text_var.set(message)
        if self.cancel_button and self.cancel_button.winfo_exists():
            self.cancel_button.pack_forget()
        if self.pause_resume_button and self.pause_resume_button.winfo_exists():
            self.pause_resume_button.pack_forget()

    def run_drawing_loop(self, commands_to_send: List[Tuple], start_index=0):
        """Sends drawing commands sequentially (RUNS IN THREAD). Handles cancel and resume."""
        total_commands = len(commands_to_send)
        
        if start_index > 0:
            self.window.after(0, lambda: self.show_drawing_progress_page(total_commands, start_index, "Resuming drawing..."))
        
        try:
            for i, (x, z, y) in enumerate(commands_to_send[start_index:], start=start_index):
                self.pause_event.wait() # This will block if the event is cleared (paused)

                if self.cancel_requested:
                    logging.info(f"Cancellation detected at command {i+1}.")
                    self._send_final_position_and_cleanup("Drawing Cancelled.", "Drawing Cancelled.")
                    return

                command_str = f"{x:.2f},{z:.2f},{y:.2f}"
                logging.debug(f"Sending command {i+1}/{total_commands}: {command_str}")

                if not self.send_message_internal(command_str):
                    logging.error(f"Connection lost while sending command {i+1}. Preparing to resume.")
                    self.resume_needed = True
                    self.resume_commands = commands_to_send
                    self.resume_start_index_global = i
                    self.last_drawing_status = {"total_commands": total_commands, "completed_commands": i, "status": "Connection Lost", "error_message": f"Lost connection before sending command {i+1}"}
                    self.window.after(0, lambda idx=i: self.update_drawing_status(idx, total_commands, "Connection Lost!"))
                    self.window.after(1000, self.connection_setup_page)
                    self.drawing_in_progress = False
                    return

                response_r = self.receive_message_internal(timeout=20.0)
                if response_r is None or response_r != "R":
                    error_msg = f"Robot did not confirm receipt (R) for command {i+1}, got '{response_r}'."
                    logging.error(error_msg + " Preparing to resume.")
                    self.resume_needed = True
                    self.resume_commands = commands_to_send
                    self.resume_start_index_global = i
                    self.last_drawing_status = {"total_commands": total_commands, "completed_commands": i, "status": f"Protocol Error (Got '{response_r}')", "error_message": error_msg}
                    self.window.after(0, lambda idx=i, r=response_r: self.update_drawing_status(idx, total_commands, f"Error: No 'R' (Got {r}). Reconnect to resume."))
                    self.window.after(1000, self.connection_setup_page)
                    self.drawing_in_progress = False
                    return

                self.window.after(0, lambda idx=i + 1: self.update_drawing_status(idx, total_commands))

            logging.info("All drawing commands sent successfully.")
            self._send_final_position_and_cleanup("Drawing Complete.", "Drawing Complete.")

        except Exception as e:
            logging.error(f"Unexpected error during drawing process: {e}", exc_info=True)
            try:
                self.window.after(0, lambda: self.update_drawing_status(i, total_commands, f"Runtime Error: {e}"))
            except (tk.TclError, NameError):
                logging.error("GUI already closed during error handling.")
            self.drawing_in_progress = False
            self.cancel_requested = False

    # --- Internal Socket Methods (without GUI popups) ---
    def send_message_internal(self, message: str) -> bool:
        """ Sends message without triggering GUI popups on error. Returns success status. """
        if not self.connected or not self.socket: return False
        try:
            self.socket.sendall(message.encode('utf-8'))
            logging.debug(f"Sent (internal): {message}")
            return True
        except (socket.error, ConnectionResetError, BrokenPipeError, socket.timeout) as e:
            logging.error(f"Send error (internal): {e}")
            self.handle_connection_loss()
            return False

    def receive_message_internal(self, timeout=20.0) -> Optional[str]:
         """ Receives message without triggering GUI popups on error. Returns message or None. """
         if not self.connected or not self.socket: return None
         try:
             self.socket.settimeout(timeout)
             data = self.socket.recv(1024)
             self.socket.settimeout(None)
             if not data:
                 logging.warning("Receive error (internal): Connection closed by peer.")
                 self.handle_connection_loss()
                 return None
             decoded_data = data.decode('utf-8').strip()
             logging.debug(f"Received (internal): {decoded_data}")
             return decoded_data
         except socket.timeout:
             logging.error(f"Timeout receiving message (internal)")
             self.handle_connection_loss()
             return None
         except (socket.error, ConnectionResetError, BrokenPipeError) as e:
             logging.error(f"Receive error (internal): {e}")
             self.handle_connection_loss()
             return None

    def handle_connection_loss(self):
        """Centralized handling of connection loss detection."""
        logging.warning("Connection lost detected.")
        was_connected = self.connected
        self.close_socket()
        # Only show popup if we were not in the middle of a drawing process
        if was_connected and not self.drawing_in_progress and not self.resume_needed:
            self.window.after(0, lambda: messagebox.showinfo("Connection Lost", "Robot connection lost."))

    # --- Connection Handling ---
    def establish_connection(self):
        """Attempt connection (modified to use internal methods and handle resume)."""
        if hasattr(self, 'connect_button') and self.connect_button.winfo_exists(): self.connect_button.config(state=tk.DISABLED)
        if hasattr(self, 'reconnect_button') and self.reconnect_button.winfo_exists(): self.reconnect_button.config(state=tk.DISABLED)

        host, port = (SIMULATION_HOST, SIMULATION_PORT) if self.connection_var.get() == "simulation" else (REAL_ROBOT_HOST, REAL_ROBOT_PORT)

        threading.Thread(target=self._connection_attempt_thread, args=(host, port), daemon=True).start()

    def _connection_attempt_thread(self, host, port):
        """Thread worker for establishing a socket connection."""
        try:
            self.close_socket()
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((host, port))
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.socket.settimeout(None)
            logging.info(f"Connected to {host}:{port}")
            self.connected = True
            self.window.after(0, lambda: self.handle_connection_result(True))
        except (socket.error, socket.timeout, ConnectionRefusedError) as e:
            logging.error(f"Connection error: {e}")
            self.connected = False
            self.close_socket()
            self.window.after(0, lambda: self.handle_connection_result(False))

    def handle_connection_result(self, connected):
        """Handle connection result and trigger resume if needed."""
        if hasattr(self, 'connect_button') and self.connect_button.winfo_exists():
            self.connect_button.config(state=tk.NORMAL)
        if hasattr(self, 'reconnect_button') and self.reconnect_button.winfo_exists():
            self.reconnect_button.config(state=tk.NORMAL)

        if connected:
            self.connection_established = True
            if self.resume_needed and self.resume_commands is not None:
                logging.info("Reconnection successful. Preparing to resume drawing.")
                self.move_to_final_before_resume()
            else:
                self.drawing_options_page()
        else:
            if self.resume_needed:
                messagebox.showerror("Reconnection Failed", "Failed to reconnect. Cannot resume the previous drawing.")
                self.resume_needed = False
                self.resume_commands = None
                self.resume_start_index_global = 0
                self.last_drawing_status["status"] = "Resume Failed"
                self.last_drawing_status["error_message"] = "Could not reconnect to robot."
                self.drawing_options_page()
            else:
                messagebox.showerror("Connection Failed", "Failed to establish connection.")

    def move_to_final_before_resume(self):
        """Sends robot to FINAL_ROBOT_POSITION and then starts resume. Runs in thread."""
        def move_and_resume_thread():
            logging.info("Moving robot to FINAL_ROBOT_POSITION before resuming...")
            self.show_drawing_progress_page(len(self.resume_commands), self.resume_start_index_global, "Moving to resume position...")

            final_x, final_z, final_y = FINAL_ROBOT_POSITION
            command_str_final = f"{final_x:.3f},{final_z:.3f},{final_y:.3f}"
            move_ok = False
            if self.connected and self.socket:
                if self.send_message_internal(command_str_final):
                    response_r = self.receive_message_internal(timeout=20.0)
                    if response_r == "R":
                        logging.info("Robot reached FINAL_ROBOT_POSITION.")
                        move_ok = True
                    else: logging.error("Failed to get 'R' confirmation for pre-resume move.")
                else: logging.error("Failed to send pre-resume move command.")

            if move_ok:
                 logging.info(f"Starting resume from command index {self.resume_start_index_global}")
                 self.drawing_in_progress = True
                 self.cancel_requested = False
                 self.pause_event.set()
                 self.run_drawing_loop(self.resume_commands, self.resume_start_index_global)
            else:
                error_msg = "Failed to move robot to safe resume position."
                logging.error(error_msg + " Cannot resume automatically, but allowing retry.")
                self.last_drawing_status["status"] = "Resume Failed (Pre-move)"
                self.last_drawing_status["error_message"] = error_msg
                self.window.after(0, lambda: messagebox.showwarning("Resume Warning", error_msg + "\nConnection might be unstable. You can try 'Reconnect & Resume' again."))
                self.drawing_in_progress = False
                self.window.after(1000, self.connection_setup_page)

        threading.Thread(target=move_and_resume_thread, daemon=True).start()

    def close_socket(self):
        """Close socket cleanly and update flags."""
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except (socket.error, OSError): pass
            finally:
                try: self.socket.close()
                except (socket.error, OSError): pass
                self.socket = None
                logging.info("Socket closed")
        self.connected = False
        self.connection_established = False

    def close_and_return_main(self):
         """Close connection and go to main page."""
         self.close_socket()
         self.resume_needed = False
         self.resume_commands = None
         self.resume_start_index_global = 0
         self.main_page()

    # --- Utility Methods ---
    def clear_frame(self):
        """Clear all widgets from the main frame."""
        if self.camera_running:
            self.stop_camera_feed()
        if hasattr(self, 'eta_update_id') and self.eta_update_id:
            self.window.after_cancel(self.eta_update_id)
            self.eta_update_id = None

        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Reset widget references
        self.camera_frame_label = None
        self.capture_button = None
        self.camera_back_button = None
        self.progress_bar = None
        self.status_label = None
        self.cancel_button = None
        self.connect_button = None
        self.reconnect_button = None
        self.preview_label = None
        self.pause_resume_button = None

    def on_window_close(self):
        """Handle window close event."""
        logging.info("Window close requested.")
        self.cancel_requested = True
        self.stop_camera_feed()
        self.close_socket()
        time.sleep(0.2)
        self.window.destroy()

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    app = RUNME_GUI()
    app.window.protocol("WM_DELETE_WINDOW", app.on_window_close)
    app.window.mainloop()
