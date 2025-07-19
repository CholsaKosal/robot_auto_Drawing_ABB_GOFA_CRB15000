# Robot Arm Auto Drawing

This repository contains a Python script that automates the process of GOFA CRB 15000 drawing any image input in the system. Please unpack pack and go file of robot studio and use the main.py script to run the program.

Rapid code of the Drawing_1 station (after unpack pack and go) must be run before running main.py

## Showcase

| H1 | H2 |
|:---|:---|
|  |  |
|  |  |


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
pip install opencv-python numpy Pillow Flask qrcode
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
