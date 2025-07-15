# Robot Image Drawing System

This project allows an ABB robot to draw images by converting them into a series of toolpath commands. The system features a Python-based GUI for image processing and a RAPID program for robot control, communicating via a highly efficient, low-latency TCP socket connection.

## How It Works

The system is composed of two main parts:

1.  **Python GUI (`main.py`):** This is the user-facing application and the "brain" of the operation. It acts as a TCP client.
    * It allows the user to either capture an image from a webcam or load one from a file.
    * It uses OpenCV to perform edge detection on the image, converting the visual lines into a set of (X, Y) coordinate vectors.
    * It processes these vectors, scaling them to fit the robot's physical workspace and optimizing the path to minimize unnecessary travel.
    * For each coordinate, it establishes the final `(X, Y, Z)` position for the robot's tool.
    * It then sends these commands to the robot one by one over a TCP socket.

2.  **RAPID Program (`AUTO_InputDrawing.mod`):** This program runs on the ABB robot controller and acts as the TCP server.
    * It listens for a connection from the Python application.
    * Once connected, it enters an infinite loop to receive and execute drawing commands.
    * For each command, it waits to receive a packet of data, executes the corresponding `MoveL` instruction, and sends back acknowledgements to the Python script to maintain synchronization.

---

## Setup and Installation

To run this project, you will need a Python environment with a few specific libraries installed. Using a virtual environment is highly recommended to keep dependencies isolated.

### 1. Create a Virtual Environment

Open a terminal or command prompt in the project's root directory and run the following command to create a virtual environment named `venv`:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

Before installing packages, you need to activate the environment.

**On Windows:**
```bash
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
source venv/bin/activate
```

Your terminal prompt should now be prefixed with `(venv)`, indicating that the virtual environment is active.

### 3. Install Required Libraries

This project requires `OpenCV` for image processing, `NumPy` for numerical operations, and `Pillow` for handling image data in the GUI. Install them using pip:

```bash
pip install opencv-python numpy Pillow
```

---

## Running the Application

1.  **Load the RAPID Program:** Load the `AUTO_InputDrawing.mod` file onto your ABB robot controller (either a real one or a virtual one in RobotStudio).
2.  **Run the RAPID Program:** Start the execution of the `main` procedure in the RAPID program. The robot will move to its home position and the program will wait for a socket connection.
3.  **Run the Python GUI:** With your virtual environment activated, run the `main.py` script from your terminal:

    ```bash
    python main.py
    ```
4.  **Connect and Draw:**
    * In the GUI, select the correct connection type (Simulation or Real Robot).
    * Click "Connect".
    * Use the test buttons to verify the connection and movement.
    * Load or capture an image, select a drawing style, and click "Confirm and Draw" to begin.

---

## The Advantage of Byte-Level Communication

For high-performance robotics, sending data as raw bytes is significantly better than higher-level methods like sending formatted strings (e.g., JSON, XML, or custom text commands).

### Why It's Better

* **Speed and Efficiency:** Raw byte communication is the fastest method possible. There is no overhead from formatting or parsing text. A floating-point number like `50.0` can be represented in just 4 bytes, whereas sending it as a string `"50.0"` takes 4 bytes plus overhead for delimiters or keys in a structured format. For an application like drawing, which can involve thousands of points, this efficiency gain is massive.
* **No Parsing Required:** When the robot controller receives a string like `"moveto(50.0, 50.0, -14.0)"`, it must spend valuable CPU cycles parsing the string: finding the numbers, converting the text characters '5', '0', '.', '0' into an actual floating-point number, and validating the format. When it receives a 12-byte packet, it can directly copy those bytes into memory and use them as three 4-byte floats. This eliminates a major source of latency.
* **Predictability and Robustness:** The protocol used in this system is rigid and predictable. Every single position command is **exactly 12 bytes**. Every acknowledgement is **exactly 1 byte**. This eliminates the synchronization problems that can occur with variable-length strings, where multiple messages can get bundled together in the TCP stream and confuse the receiver. By enforcing a strict data size, we ensure the client and server always stay in sync.

### How This System Uses It

1.  **Packing in Python:** The `main.py` script takes three Python float numbers for the `(X, Y, Z)` coordinates. It uses the `struct.pack('!fff', x, y, z)` function to convert them into a 12-byte binary representation.
    * `!`: Specifies that the data should be in network byte order (big-endian), which is a universal standard.
    * `fff`: Specifies that the data consists of three consecutive 4-byte floating-point numbers.
2.  **Sending and Receiving:** This 12-byte packet is sent over the TCP socket. The RAPID program uses `SocketReceive` with the `\ReadNoOfBytes:=12` argument, ensuring it reads the complete packet and nothing more.
3.  **Unpacking in RAPID:** The RAPID program uses the `UnpackRawBytes` instruction. Critically, it manually advances a `read_pos` pointer after each value is read, ensuring it unpacks the X, Y, and Z values from the correct locations within the 12-byte packet. This allows it to instantly reconstruct the three floating-point numbers from the raw data.

This tightly-coupled, efficient protocol is what allows the system to stream a high volume of coordinates to the robot, enabling smooth and continuous drawing motions.
