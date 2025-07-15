import cv2
import numpy as np
import socket
import time
import math # Import math for distance calculation

# --- Constants ---
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
PEN_UP_Z = -10  # Z coordinate for pen up (adjust based on your robot setup)
PEN_DOWN_Z = 0    # Z coordinate for pen down (adjust based on your robot setup)
MIN_CONTOUR_LENGTH_PX = 10 # Ignore very small contours (noise) - adjust as needed

# --- Helper Function ---
def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x, y)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- Image Processing Functions ---
def image_to_contours(image_path, output_path, threshold1=25, threshold2=100):
    """
    Convert an image to edges using Canny and find contours.
    :param image_path: Path to the input image.
    :param output_path: Path to save the edge image (optional visual check).
    :param threshold1: Lower threshold for Canny edge detection.
    :param threshold2: Upper threshold for Canny edge detection.
    :return: List of contours (each a list of pixel coordinates), image dimensions (width, height).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    image_height, image_width = image.shape[:2] # Get image dimensions

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1, threshold2)
    cv2.imwrite(output_path, edges) # Save the edges image for verification

    # Find contours - Use RETR_LIST to get all contours, CHAIN_APPROX_SIMPLE to compress segments
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out very small contours (likely noise)
    filtered_contours = [c for c in contours if cv2.arcLength(c, closed=False) > MIN_CONTOUR_LENGTH_PX]

    print(f"Found {len(contours)} raw contours, {len(filtered_contours)} contours after filtering.")

    # Reshape contours to be lists of (x, y) tuples
    contours_xy = []
    for contour in filtered_contours:
        # Squeeze removes single-dimensional entries from the shape of an array.
        # Contours from findContours are often [[[x1, y1]], [[x2, y2]], ...]
        points = contour.squeeze().tolist()
        # Handle cases where squeeze might result in a single point (not a list)
        if isinstance(points[0], int):
             points = [points] # Make it a list containing the single point tuple
        contours_xy.append([(p[0], p[1]) for p in points])


    return contours_xy, image_width, image_height

def scale_point_to_a4(point_xy, image_width, image_height, scale_factor):
    """
    Scales and transforms a single (x, y) pixel coordinate to centered A4 (mm).
    :param point_xy: Tuple (x, y) pixel coordinate.
    :param image_width: Width of the original image in pixels.
    :param image_height: Height of the original image in pixels.
    :param scale_factor: Pre-calculated scale factor for A4 fitting.
    :return: Tuple (x_mm, y_mm) scaled coordinate.
    """
    x_pixel, y_pixel = point_xy
    center_x_pixel = image_width / 2
    center_y_pixel = image_height / 2

    # Translate pixel coordinates so (0,0) is the image center
    x_centered_pixel = x_pixel - center_x_pixel
    # Invert y-axis (image y increases downwards, cartesian y increases upwards)
    y_centered_pixel = center_y_pixel - y_pixel

    # Scale the centered coordinates to millimeters
    x_mm = x_centered_pixel * scale_factor
    y_mm = y_centered_pixel * scale_factor

    return (x_mm, y_mm)

def create_drawing_paths(contours_xy, image_width, image_height, optimize_paths=True):
    """
    Takes list of contours (pixel coordinates), scales them, and creates drawing paths.
    Each path is a sequence of (X, Z, Y) commands for the robot.
    :param contours_xy: List of contours, where each contour is a list of (x, y) pixel points.
    :param image_width: Width of the original image.
    :param image_height: Height of the original image.
    :param optimize_paths: Boolean flag to enable path optimization (nearest neighbor).
    :return: List of robot commands [(X, Z, Y), ...].
    """
    # Calculate the single scale factor for the entire drawing
    scale_x = A4_WIDTH_MM / image_width
    scale_y = A4_HEIGHT_MM / image_height
    scale_factor = min(scale_x, scale_y)

    scaled_contours = []
    for contour in contours_xy:
        scaled_contour = [scale_point_to_a4(p, image_width, image_height, scale_factor) for p in contour]
        if len(scaled_contour) >= 2: # Need at least two points to form a line
             scaled_contours.append(scaled_contour)

    if not scaled_contours:
        return []

    # --- Path Optimization (Optional but Recommended) ---
    # Order the contours to minimize travel distance between the end of one contour
    # and the start of the next (Nearest Neighbor approach).
    ordered_contours = []
    if optimize_paths and scaled_contours:
        remaining_contours = list(scaled_contours) # Make a copy
        # Start with the first contour (arbitrary, could pick longest/closest to origin etc.)
        current_contour = remaining_contours.pop(0)
        ordered_contours.append(current_contour)
        last_point = current_contour[-1] # End point of the current contour

        while remaining_contours:
            best_dist = float('inf')
            best_idx = -1
            best_reversed = False # Should we draw the next contour in reverse?

            for i, contour in enumerate(remaining_contours):
                start_point = contour[0]
                end_point = contour[-1]

                # Distance from last_point to start of this contour
                dist_start = calculate_distance(last_point, start_point)
                # Distance from last_point to end of this contour (if drawn in reverse)
                dist_end = calculate_distance(last_point, end_point)

                if dist_start < best_dist:
                    best_dist = dist_start
                    best_idx = i
                    best_reversed = False

                if dist_end < best_dist:
                    best_dist = dist_end
                    best_idx = i
                    best_reversed = True

            # Select the best next contour
            next_contour = remaining_contours.pop(best_idx)
            if best_reversed:
                next_contour.reverse() # Reverse it in place

            ordered_contours.append(next_contour)
            last_point = next_contour[-1] # Update last point for next iteration

        scaled_contours = ordered_contours # Use the optimized order
        print(f"Optimized contour drawing order.")

    # --- Generate Robot Commands ---
    robot_commands = []
    for contour in scaled_contours:
        start_point = contour[0]
        # 1. Move to the start of the contour (Pen Up)
        robot_commands.append((start_point[0], PEN_UP_Z, start_point[1]))
        # 2. Put Pen Down at the start
        robot_commands.append((start_point[0], PEN_DOWN_Z, start_point[1]))

        # 3. Draw along the contour segments (Pen Down)
        for i in range(len(contour) - 1):
            end_point = contour[i+1]
            # Only add command if it's different from the previous (handles single points)
            if end_point != contour[i]:
                 robot_commands.append((end_point[0], PEN_DOWN_Z, end_point[1]))

        # 4. Lift Pen at the end of the contour
        final_point = contour[-1]
        robot_commands.append((final_point[0], PEN_UP_Z, final_point[1]))

    return robot_commands


# --- Socket Communication Functions ---
# (connect_to_robot, send_message, receive_message remain the same)
def connect_to_robot(host, port):
    """ Connect to the robot controller and return the socket object. """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((host, port))
        print(f"Connected to the robot controller at {host}:{port}")
        return s
    except socket.timeout:
        print("Socket timeout during connection")
        return None
    except socket.error as e:
        print(f"Socket error during connection: {e}")
        return None

def send_message(s, message):
    """ Send a message to the robot controller. """
    try:
        s.sendall(message.encode('utf-8'))
        # print(f"Sent: {message}") # Reduce verbosity
        return True
    except socket.timeout:
        print("Socket timeout during send")
        return False
    except socket.error as e:
        print(f"Send error: {e}")
        return False

def receive_message(s, buffer_size=1024):
    """ Receive a message from the robot controller. """
    try:
        data = s.recv(buffer_size)
        decoded_data = data.decode('utf-8').strip()
        # print(f"Received: {decoded_data}") # Reduce verbosity
        return decoded_data
    except socket.timeout:
        print("Socket timeout during receive")
        return None
    except socket.error as e:
        print(f"Receive error: {e}")
        return None

# --- Function to dump point data into a text file ---
def dump_path_data(robot_commands, file_path):
    """
    Dump the robot command path data into a text file.
    Format: X, Z, Y
    :param robot_commands: List of robot commands [(X, Z, Y), ...].
    :param file_path: Path to save the point data file.
    """
    with open(file_path, 'w') as file:
        for (x, z, y) in robot_commands:
            file.write(f"{x:.2f}, {z}, {y:.2f}\n")


# --- Main Function ---
def main():
    input_image_path = "C:/Users/USER/Learn_Coding/TROB1_drawing_with_voice_command/image2.jpg"
    output_edge_path = "C:/Users/USER/Learn_Coding/TROB1_drawing_with_voice_command/line_art_edges_output.jpg"
    output_path_data_path = "C:/Users/USER/Learn_Coding/TROB1_drawing_with_voice_command/path_data_a4_optimized.txt"

    HOST = '127.0.0.1'
    PORT = 55000

    try:
        # Step 1: Find contours from the image
        contours_xy, image_width, image_height = image_to_contours(input_image_path, output_edge_path)

        if not contours_xy:
            print("No contours found or remaining after filtering.")
            return

        # Step 2: Create scaled drawing paths from contours
        # optimize_paths=True will try to draw contours in an efficient order
        robot_commands = create_drawing_paths(contours_xy, image_width, image_height, optimize_paths=True)
        print(f"Generated {len(robot_commands)} robot commands.")

        if not robot_commands:
             print("No robot commands generated.")
             return

        # Step 3: Dump the command path data to a file
        dump_path_data(robot_commands, output_path_data_path)
        print(f"Optimized path data saved to {output_path_data_path}")

        # Step 4: Socket communication
        user_input = input("Enter '1' to connect to the robot and send drawing instructions: ")
        if user_input == '1':
            robot_socket = connect_to_robot(HOST, PORT)
            if robot_socket is None:
                print("Failed to connect.")
                return

            try:
                total_commands = len(robot_commands)
                for i, (x, z, y) in enumerate(robot_commands):
                    command_str = f"{x:.2f},{z},{y:.2f}"
                    print(f"Sending command {i+1}/{total_commands}: {command_str} ", end="")

                    if not send_message(robot_socket, command_str):
                        print(" Send failed.")
                        break

                    # Wait for receipt (R)
                    response_r = receive_message(robot_socket)
                    if response_r != "R":
                        print(f" Robot did not confirm receipt (R), got '{response_r}'. Stopping.")
                        break
                    print("[R] ", end="")

                    # Wait for completion (D)
                    response_d = receive_message(robot_socket)
                    if response_d != "D":
                        print(f" Robot did not confirm completion (D), got '{response_d}'. Stopping.")
                        break
                    print("[D]")

                    # Optional short delay
                    # time.sleep(0.05)

                print("\n--- Drawing sequence finished or stopped ---")

            except Exception as e:
                print(f"\nAn error occurred during robot communication: {e}")
            finally:
                robot_socket.close()
                print("Connection closed")
        else:
            print("Connection aborted by user.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()