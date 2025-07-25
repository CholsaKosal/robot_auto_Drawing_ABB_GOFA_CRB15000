# testing/main.py
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import os
import threading
import time
import logging
import socket
import struct # Import the struct module for packing data
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import cv2
import numpy as np
import math
from PIL import Image, ImageTk

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
# Standardized to (X, Y, Z)
FINAL_ROBOT_POSITION = (0, 0, -350)
A4_WIDTH_MM = 180
A4_HEIGHT_MM = 217
# This is now a DEFAULT value, which can be overridden by the GUI.
PEN_DOWN_Z = -14 
PEN_UP_Z = 1.3 * PEN_DOWN_Z 
MIN_CONTOUR_LENGTH_PX = 10

# Threshold options
THRESHOLD_OPTIONS = [
    ("Option {}".format(i), i*10, i*20) for i in range(1, 8)
]

# Time estimation factor
TIME_ESTIMATE_FACTOR = 0.018

# Waypoints for the signature drawing, using specific Z-depths.
SIGNATURE_WAYPOINTS = ((0, 0, -70),)


def create_signature_commands(waypoints, pen_down_z):
    """Converts signature waypoints into robot commands, handling pen up/down moves."""
    pen_up_z = 1.3 * pen_down_z
    if pen_up_z == 0:
        pen_up_z = -3
    commands = []
    if not waypoints:
        return commands

    # Move to the starting XY of the signature with the pen up
    start_x, start_y, _ = waypoints[0]
    commands.append((start_x, start_y, pen_up_z))

    # Add the actual drawing waypoints
    for point in waypoints:
        commands.append(point)

    # Lift the pen at the end of the signature
    if commands:
        last_x, last_y, _ = waypoints[-1]
        commands.append((last_x, last_y, pen_up_z))
    return commands


def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x, y)."""
    if p1 is None or p2 is None: return float('inf')
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def image_to_contours_internal(image_path_or_array, threshold1, threshold2, save_edge_path=None):
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_path_or_array, np.ndarray):
        if len(image_path_or_array.shape) == 3:
            image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY)
        else:
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

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.arcLength(c, closed=False) > MIN_CONTOUR_LENGTH_PX]

    contours_xy = []
    for contour in filtered_contours:
        points = contour.squeeze().tolist()
        if not isinstance(points, list) or not points: continue
        if isinstance(points[0], int):
            points = [points]
        contours_xy.append([(p[0], p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) == 2])

    contours_xy = [c for c in contours_xy if c]
    return contours_xy, image_width, image_height


def scale_point_to_a4(point_xy, image_width, image_height, scale_factor):
    x_pixel, y_pixel = point_xy
    center_x_pixel = image_width / 2
    center_y_pixel = image_height / 2
    x_centered_pixel = x_pixel - center_x_pixel
    y_centered_pixel = center_y_pixel - y_pixel
    x_mm = x_centered_pixel * scale_factor
    y_mm = y_centered_pixel * scale_factor
    return (x_mm, y_mm)


def create_drawing_paths(contours_xy, image_width, image_height, pen_down_z, optimize_paths=True):
    """Generates robot drawing paths using a specific Z-height for pen-down moves."""
    if not contours_xy or image_width <= 0 or image_height <= 0:
        return []

    pen_up_z = 1.3 * pen_down_z
    if pen_up_z == 0:
        pen_up_z = -3

    scale_x = A4_WIDTH_MM / image_width
    scale_y = A4_HEIGHT_MM / image_height
    scale_factor = min(scale_x, scale_y)

    scaled_contours = []
    for contour in contours_xy:
        if not contour: continue
        scaled_contour = [scale_point_to_a4(p, image_width, image_height, scale_factor) for p in contour]
        if len(scaled_contour) >= 2:
            scaled_contours.append(scaled_contour)
        elif len(scaled_contour) == 1:
            scaled_contours.append([scaled_contour[0], scaled_contour[0]])

    if not scaled_contours:
        return []

    ordered_contours = []
    last_point = None
    if optimize_paths:
        remaining_contours = list(scaled_contours)
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
                    if dist_end < best_dist:
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
                    logging.warning("Path optimization loop finished unexpectedly.")
                    break
        scaled_contours = ordered_contours
    else:
        scaled_contours = [c for c in scaled_contours]

    robot_commands = []
    for contour in scaled_contours:
        if not contour: continue
        start_point = contour[0]
        # Standardized to (X, Y, Z)
        robot_commands.append((start_point[0], start_point[1], pen_up_z))
        robot_commands.append((start_point[0], start_point[1], pen_down_z))

        for i in range(len(contour) - 1):
            end_point = contour[i+1]
            if end_point != contour[i]:
                # Standardized to (X, Y, Z)
                robot_commands.append((end_point[0], end_point[1], pen_down_z))

        final_point = contour[-1]
        # Standardized to (X, Y, Z)
        robot_commands.append((final_point[0], final_point[1], pen_up_z))

    return robot_commands


class RUNME_GUI:
    """Main GUI application for the Robotics System."""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Robotics Drawing GUI")
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.connection_var = tk.StringVar(value="simulation")
        self.socket = None
        self.connected = False
        self.connection_established = False

        self.cap = None
        self.camera_running = False
        self.camera_frame_label = None
        self.capture_button = None
        self.camera_back_button = None

        self.current_image_path = None
        self.threshold_options_data = {}
        self.selected_commands = None
        self.drawing_in_progress = False
        self.cancel_requested = False
        self.progress_bar = None
        self.status_label = None
        self.cancel_button = None
        self.reconnect_button = None
        
        self.test_z_var = tk.StringVar(value=str(PEN_DOWN_Z))
        
        self.pause_event = threading.Event()
        self.pause_resume_button = None

        self.last_drawing_status = {
            "total_commands": 0,
            "completed_commands": 0,
            "status": "None",
            "error_message": ""
        }
        
        self.resume_needed = False
        self.resume_commands = None
        self.resume_total_original_commands = 0
        self.resume_start_index_global = 0

        # --- ETA Countdown variables ---
        self.eta_update_id = None
        self.drawing_start_time = 0
        self.total_paused_time = 0
        self.pause_start_time = 0
        self.progress_text_var = tk.StringVar()


        self.main_page()

    def main_page(self):
        self.clear_frame()
        tk.Label(self.main_frame, text="Robotics Drawing System", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.main_frame, text="Setup Connection & Draw",
                  command=self.connection_setup_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Exit",
                  command=self.on_window_close, width=30).pack(pady=5)

    def connection_setup_page(self):
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

        test_z_frame = tk.Frame(self.main_frame, pady=5)
        test_z_frame.pack(pady=5)

        tk.Label(test_z_frame, text="Test / Drawing Z-Coordinate:").pack(side=tk.LEFT, padx=5)
        test_z_entry = tk.Entry(test_z_frame, textvariable=self.test_z_var, width=10)
        test_z_entry.pack(side=tk.LEFT)
        
        self.send_z_button = tk.Button(test_z_frame, text="Send to (0, 0, Z)", command=self.send_to_test_z_action)
        self.send_z_button.pack(side=tk.LEFT, padx=10)

        test_buttons_frame = tk.Frame(self.main_frame)
        test_buttons_frame.pack(pady=5)

        self.safe_center_button = tk.Button(test_buttons_frame, text="Go to Safe Center", command=self.send_to_safe_center_action, width=20)
        self.safe_center_button.pack(side=tk.LEFT, padx=5)

        self.test_workspace_button = tk.Button(test_buttons_frame, text="Test Workspace", command=self.test_workspace_action, width=20)
        self.test_workspace_button.pack(side=tk.LEFT, padx=5)

        tk.Button(self.main_frame, text="Capture Image to Draw",
                  command=self.capture_image_page, width=30).pack(pady=5)
        tk.Button(self.main_frame, text="Input Image to Draw",
                  command=self.input_image_page, width=30).pack(pady=5)
        
        tk.Button(self.main_frame, text="Disconnect",
                  command=self.close_and_return_main, width=30).pack(pady=5)

    def send_to_test_z_action(self):
        try:
            test_z = float(self.test_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Z-coordinate must be a valid number.")
            return

        if self.send_z_button and self.send_z_button.winfo_exists():
            self.send_z_button.config(state=tk.DISABLED)
        # Standardized to (X, Y, Z)
        threading.Thread(target=self._send_command_sequence_thread, args=([(0.0, 0.0, test_z)], self.send_z_button), daemon=True).start()

    def send_to_safe_center_action(self):
        safe_z = 6 * PEN_DOWN_Z
        
        if hasattr(self, 'safe_center_button') and self.safe_center_button.winfo_exists():
            self.safe_center_button.config(state=tk.DISABLED)
        
        # Standardized to (X, Y, Z)
        logging.info(f"Sending robot to safe center (0, 0, {safe_z})")
        threading.Thread(target=self._send_command_sequence_thread, args=([(0, 0, safe_z)], self.safe_center_button), daemon=True).start()

    def test_workspace_action(self):
        """Sends the robot on a path to outline the workspace corners."""
        try:
            test_z = float(self.test_z_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "The Z-coordinate for testing must be a valid number.")
            return

        if hasattr(self, 'test_workspace_button') and self.test_workspace_button.winfo_exists():
            self.test_workspace_button.config(state=tk.DISABLED)

        pen_up_z = 1.3 * test_z
        
        if pen_up_z == 0:
            pen_up_z = -3

        # Standardized to (X, Y, Z)
        workspace_path = [
            (0,0, test_z),
            (50, 50, test_z),
            (50, -50, test_z),
            (-50, -50, test_z),
            (-50, 50, test_z),
            (0,0, test_z),
            (0, 0, pen_up_z)
        ]
        
        logging.info("Starting workspace test...")
        
        threading.Thread(target=self._send_command_sequence_thread, args=(workspace_path, self.test_workspace_button), daemon=True).start()

    def _send_command_sequence_thread(self, commands: List[Tuple], button_to_re_enable: tk.Button):
        """Thread worker to send a sequence of commands, one by one, using the new byte protocol."""
        original_text = button_to_re_enable.cget("text")
        self.window.after(0, lambda: button_to_re_enable.config(text="Moving..."))

        # Standardized to (X, Y, Z)
        for i, (x, y, z) in enumerate(commands):
            if self.cancel_requested:
                logging.info("Test sequence cancelled.")
                break
            
            logging.info(f"Sending command {i+1}/{len(commands)}: ({x:.2f}, {y:.2f}, {z:.2f})")
            # Standardized to pack (X, Y, Z)
            byte_data = struct.pack('!fff', x, y, z)
            
            if self.send_message_internal(byte_data):
                response_r = self.receive_message_internal(timeout=5.0)
                if response_r == "R":
                    logging.info("Received 'R' (Ready) from robot.")
                    response_d = self.receive_message_internal(timeout=None)
                    # if response_d == "D":
                    #     logging.info("Received 'D' (Done) from robot.")
                    # else:
                    #     error_msg = f"Robot did not confirm completion (D) for command {i+1}. Got: '{response_d}'"
                    #     logging.error(error_msg)
                    #     self.window.after(0, lambda: messagebox.showerror("Test Failed", error_msg))
                    #     break
                else:
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


    def capture_image_page(self):
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

        self.window.bind('s', self.capture_action_event)
        self.window.bind('S', self.capture_action_event)

        self.start_camera_feed()

    def start_camera_feed(self):
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
        if not self.camera_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)

            if self.camera_frame_label:
                self.camera_frame_label.imgtk = imgtk
                self.camera_frame_label.configure(image=imgtk)
        else:
            logging.warning("Failed to grab frame from camera.")

        if self.camera_running:
            self.window.after(30, self._update_camera_frame)

    def stop_camera_feed(self):
        self.camera_running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
            self.cap = None

    def stop_camera_and_go_back(self):
        self.stop_camera_feed()
        self.window.unbind('s')
        self.window.unbind('S')
        self.drawing_options_page()

    def capture_action_event(self, event=None):
        self.capture_action()

    def capture_action(self):
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

    def input_image_page(self):
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
        filepath = filedialog.askopenfilename(
            title="Select Image to Draw",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")]
        )
        if filepath:
            self.image_path_var.set(filepath)

    def process_input_image(self):
        filepath = self.image_path_var.get()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror("Error", f"Invalid or non-existent file path:\n{filepath}")
            return
        self.current_image_path = filepath
        self.show_threshold_options(self.current_image_path)

    def show_threshold_options(self, image_path):
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

        try:
            pen_down_z = float(self.test_z_var.get())
            logging.info(f"Using custom pen down Z for drawing path generation: {pen_down_z}")
        except (ValueError, tk.TclError):
            logging.warning("Invalid or no custom Z value found, using default.")
            pen_down_z = PEN_DOWN_Z

        threading.Thread(target=self._process_threshold_options_thread, args=(image_path, options_frame, loading_label, pen_down_z), daemon=True).start()

    def _process_threshold_options_thread(self, image_path, options_frame, loading_label, pen_down_z):
        results = {}
        preview_paths = {}

        for i, (label, t1, t2) in enumerate(THRESHOLD_OPTIONS):
            logging.info(f"Processing option: {label} (t1={t1}, t2={t2})")
            preview_path = TMP_EDGE_OUTPUT_PATH.format(i)
            contours_xy, w, h = image_to_contours_internal(image_path, t1, t2, save_edge_path=preview_path)

            if contours_xy is None or w == 0 or h == 0:
                logging.warning(f"Failed to process contours for option {label}")
                results[label] = None
                preview_paths[label] = None
                continue

            commands = create_drawing_paths(contours_xy, w, h, pen_down_z, optimize_paths=True)
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
                results[label] = None
                preview_paths[label] = None
                logging.warning(f"No commands generated for option {label}")

        self.window.after(0, lambda: self._display_threshold_options(options_frame, loading_label, results, preview_paths))

    def _display_threshold_options(self, options_frame, loading_label, results, preview_paths):
        loading_label.destroy()

        self.threshold_options_data = results
        self.edge_preview_paths = preview_paths

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
            # User cancelled the save dialog
            return

        try:
            with open(filepath, 'w') as f:
                # Format as a tuple of tuples: ((x,y,z),(x,y,z),...)
                points_str = ",".join([f"({x:.2f},{y:.2f},{z:.2f})" for x, y, z in commands])
                f.write(f"({points_str})")
            
            messagebox.showinfo("Success", f"Drawing points successfully saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save the file.\nError: {e}")

    def show_edge_preview(self, option_label):
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
            self.pause_event.set()
            
            try:
                pen_down_z = float(self.test_z_var.get())
            except (ValueError, tk.TclError):
                pen_down_z = PEN_DOWN_Z

            full_command_list = self.selected_commands + create_signature_commands(SIGNATURE_WAYPOINTS, pen_down_z)
            
            # --- ETA Calculation ---
            self.total_estimated_time = len(full_command_list) * TIME_ESTIMATE_FACTOR
            self.drawing_start_time = time.time()
            self.total_paused_time = 0
            self.pause_start_time = 0
            
            threading.Thread(target=self.run_drawing_loop, args=(full_command_list,), daemon=True).start()
            self.show_drawing_progress_page(len(full_command_list))
        else:
            messagebox.showwarning("Busy", "Drawing already in progress.")

    def show_drawing_progress_page(self, total_commands, current_progress=0):
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

        # Start the ETA update loop
        self.update_drawing_status(current_progress, total_commands)
        self._update_eta_countdown()


    def _update_eta_countdown(self):
        """Periodically updates the ETA label with a dynamic estimate."""
        if not self.drawing_in_progress:
            return

        completed_cmds = self.progress_bar['value']
        total_cmds = self.progress_bar['maximum']
        
        remaining_time = 0
        
        # Only calculate dynamic ETA after a few commands have completed for stability
        if completed_cmds > 5:
            # Calculate total time spent actively drawing (excluding pauses)
            active_drawing_time = (time.time() - self.drawing_start_time) - self.total_paused_time
            avg_time_per_cmd = active_drawing_time / completed_cmds
            remaining_cmds = total_cmds - completed_cmds
            remaining_time = remaining_cmds * avg_time_per_cmd
        else:
            # For the beginning, use the initial static estimate
            elapsed_time = (time.time() - self.drawing_start_time) - self.total_paused_time
            initial_total_time = total_cmds * TIME_ESTIMATE_FACTOR
            remaining_time = initial_total_time - elapsed_time

        if remaining_time < 0:
            remaining_time = 0

        mins, secs = divmod(int(remaining_time), 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        # Update the full progress text
        self.progress_text_var.set(f"Sent {completed_cmds} / {total_cmds} commands | ETA: {time_str}")

        # Schedule the next update
        self.eta_update_id = self.window.after(1000, self._update_eta_countdown)


    def toggle_pause_resume(self):
        if self.pause_event.is_set():
            # --- PAUSING ---
            self.pause_event.clear()
            logging.info("Drawing paused by user.")
            if self.pause_resume_button and self.pause_resume_button.winfo_exists():
                self.pause_resume_button.config(text="Resume")
            
            # Record when the pause started
            self.pause_start_time = time.time()
            
            # Stop the timer when paused
            if self.eta_update_id:
                self.window.after_cancel(self.eta_update_id)
                self.eta_update_id = None
        else:
            # --- RESUMING ---
            self.pause_event.set()
            logging.info("Drawing resumed by user.")
            if self.pause_resume_button and self.pause_resume_button.winfo_exists():
                self.pause_resume_button.config(text="Pause")
            
            # Add the duration of the pause to the total paused time
            if self.pause_start_time > 0:
                paused_duration = time.time() - self.pause_start_time
                self.total_paused_time += paused_duration
                self.pause_start_time = 0 # Reset for next pause

            # Restart the timer when resumed
            self._update_eta_countdown()

    def update_drawing_status(self, current_command_index, total_commands):
        """Updates the progress bar. The label is handled by the ETA loop."""
        if self.progress_bar and self.progress_bar.winfo_exists():
            self.progress_bar['value'] = current_command_index
        
        # The text variable is now updated entirely by the _update_eta_countdown loop
        # to prevent race conditions and keep the display smooth.

    def request_cancel_drawing(self):
        if self.drawing_in_progress:
            logging.info("Cancel requested by user.")
            self.cancel_requested = True
            self.pause_event.set() # Unblock the drawing loop so it can see the flag
            if self.cancel_button and self.cancel_button.winfo_exists():
                self.cancel_button.config(text="Cancelling...", state=tk.DISABLED)
            if self.pause_resume_button and self.pause_resume_button.winfo_exists():
                self.pause_resume_button.config(state=tk.DISABLED)
            if self.status_label and self.status_label.winfo_exists():
                self.progress_text_var.set("Cancellation requested...")

    def _send_final_position_and_cleanup(self, success_message, failure_message):
        self.drawing_in_progress = False # Stop the ETA loop
        logging.info("Attempting to move robot to final position.")
        # Standardized to (X, Y, Z)
        final_x, final_y, final_z = FINAL_ROBOT_POSITION
        byte_data = struct.pack('!fff', final_x, final_y, final_z)

        move_ok = False
        if self.connected and self.socket:
            if self.send_message_internal(byte_data):
                response_r_final = self.receive_message_internal(timeout=None)
                if response_r_final == "R":
                    response_d_final = self.receive_message_internal(timeout=None)
                    # if response_d_final == "D":
                    #     logging.info("Robot reached final position.")
                    #     move_ok = True
                    # else:
                    #     logging.error(f"Robot didn't confirm final move completion (D), got '{response_d_final}'")
                else:
                    logging.error(f"Robot didn't confirm final move receipt (R), got '{response_r_final}'")
            else:
                logging.error("Failed to send final position command.")

        final_status = ""
        if move_ok:
            final_status = f"{success_message} Robot at final position."
        else:
            final_status = f"{failure_message} Failed to reach final position."

        self.last_drawing_status["status"] = success_message
        self.last_drawing_status["error_message"] = "" if move_ok else "Failed to reach final position."

        self.window.after(0, lambda fs=final_status: self.update_final_status(fs))

        self.selected_commands = None
        self.cancel_requested = False
        if not self.resume_needed:
            self.resume_commands = None
            self.resume_total_original_commands = 0
            self.resume_start_index_global = 0

        self.window.after(2000, self.drawing_options_page)

    def update_final_status(self, message):
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
        """Sends drawing commands one by one using the byte protocol."""
        total_commands = len(commands_to_send)
        
        if start_index > 0:
            self.window.after(0, lambda: self.show_drawing_progress_page(total_commands, start_index))
        
        try:
            # Standardized to (X, Y, Z)
            for i, (x, y, z) in enumerate(commands_to_send[start_index:]):
                self.pause_event.wait()
                if self.cancel_requested:
                    logging.info("Cancellation detected in drawing loop.")
                    # Call the cleanup function to move to a final safe position and reset the UI
                    self._send_final_position_and_cleanup("Drawing Cancelled.", "Drawing Cancelled.")
                    return

                current_command_global_index = start_index + i
                
                # Standardized to pack (X, Y, Z)
                byte_data = struct.pack('!fff', x, y, z)

                if not self.send_message_internal(byte_data):
                    logging.error(f"Connection lost while sending command. Preparing to resume.")
                    self.drawing_in_progress = False # Stop ETA loop
                    self.resume_needed = True
                    self.resume_commands = commands_to_send
                    self.resume_start_index_global = current_command_global_index
                    return

                response_r = self.receive_message_internal(timeout=10.0)
                if response_r != "R":
                    logging.error(f"Robot did not confirm receipt (R), got '{response_r}'.")
                    self.drawing_in_progress = False
                    return
                
                response_d = self.receive_message_internal(timeout=None)
                # if response_d != "D":
                #     logging.error(f"Robot did not confirm completion (D), got '{response_d}'.")
                #     self.drawing_in_progress = False
                #     return
                
                # Update the progress bar value. The label text is handled by the ETA loop.
                self.window.after(0, lambda p=current_command_global_index + 1: self.update_drawing_status(p, total_commands))

            logging.info("All drawing commands sent successfully.")
            self._send_final_position_and_cleanup("Drawing Complete.", "Drawing Complete.")

        except Exception as e:
            logging.error(f"Unexpected error during drawing process: {e}", exc_info=True)
            self.drawing_in_progress = False

    def send_message_internal(self, message: bytes) -> bool:
        """ Sends byte data without triggering GUI popups on error. Returns success status. """
        if not self.connected or not self.socket: return False
        try:
            self.socket.sendall(message)
            return True
        except (socket.error, ConnectionResetError, BrokenPipeError, socket.timeout) as e:
            logging.error(f"Send error (internal): {e}")
            self.handle_connection_loss()
            return False

    def receive_message_internal(self, timeout=None) -> Optional[str]:
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
            return decoded_data
        except socket.timeout:
            logging.error(f"Timeout receiving message (internal)")
            return None
        except (socket.error, ConnectionResetError, BrokenPipeError) as e:
            logging.error(f"Receive error (internal): {e}")
            self.handle_connection_loss()
            return None

    def handle_connection_loss(self):
        logging.warning("Connection lost detected.")
        was_connected = self.connected
        self.close_socket()
        if was_connected and not self.drawing_in_progress and not self.resume_needed:
            self.window.after(0, lambda: messagebox.showinfo("Connection Lost", "Robot connection lost."))

    def establish_connection(self):
        if hasattr(self, 'connect_button') and self.connect_button.winfo_exists(): self.connect_button.config(state=tk.DISABLED)
        if hasattr(self, 'reconnect_button') and self.reconnect_button.winfo_exists(): self.reconnect_button.config(state=tk.DISABLED)

        host, port = (SIMULATION_HOST, SIMULATION_PORT) if self.connection_var.get() == "simulation" else (REAL_ROBOT_HOST, REAL_ROBOT_PORT)

        def connection_attempt():
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
        threading.Thread(target=connection_attempt, daemon=True).start()

    def handle_connection_result(self, connected):
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
                self.resume_total_original_commands = 0
                self.resume_start_index_global = 0
                self.last_drawing_status["status"] = "Resume Failed"
                self.last_drawing_status["error_message"] = "Could not reconnect to robot."
                self.drawing_options_page()
            else:
                messagebox.showerror("Connection Failed", "Failed to establish connection.")

    def move_to_final_before_resume(self):
        def move_and_resume_thread():
            logging.info("Moving robot to FINAL_ROBOT_POSITION before resuming...")
            self.show_drawing_progress_page(self.resume_total_original_commands, self.resume_start_index_global)

            # Standardized to (X, Y, Z)
            final_x, final_y, final_z = FINAL_ROBOT_POSITION
            byte_data = struct.pack('!fff', final_x, final_y, final_z)
            move_ok = False
            if self.connected and self.socket:
                if self.send_message_internal(byte_data):
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
        self.pause_event.set()
        self.close_socket()
        self.resume_needed = False
        self.resume_commands = None
        self.resume_total_original_commands = 0
        self.resume_start_index_global = 0
        self.main_page()

    def clear_frame(self):
        if self.camera_running:
            self.stop_camera_feed()
        # Cancel any pending ETA update
        if self.eta_update_id:
            self.window.after_cancel(self.eta_update_id)
            self.eta_update_id = None
        for widget in self.main_frame.winfo_children():
            widget.destroy()
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
        logging.info("Window close requested.")
        self.cancel_requested = True
        self.drawing_in_progress = False # Stop ETA loop
        self.pause_event.set()
        self.stop_camera_feed()
        self.close_socket()
        time.sleep(0.2)
        self.window.destroy()

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    app = RUNME_GUI()
    app.window.protocol("WM_DELETE_WINDOW", app.on_window_close)
    app.window.mainloop()
