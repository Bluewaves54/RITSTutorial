
# Phase 1: Setup Raspberry Pi

## Hardware setup

**Materials - if any of the following are missing, please ask an instructor:**
- Encased Raspberry Pi 5 w/AI HAT+
- Raspberry PI USB-C Power Supply
- Monitor, mouse, and keyboard
- HDMI to mini-HDMI cable
- PiCamera module + 3d printed case

Carefully remove white lid on top of Raspberry Pi

Plug mini-HDMI to the correct port on the side of the Pi and plug the HDMI to the monitor

Plug in mouse and keyboard into any of the USB ports on the front side of the Pi

### PiCamera installation:
- locate the Camera Serial Interface (CSI) port on your Raspberry Pi
- Gently pull up the tabs on the top edge of the CSI connector. This will release the locking mechanism
- Ensure the ribbon cable is oriented correctly. The shiny metal contacts on the cable should face away from the Ethernet port (or towards the HDMI port).
- Carefully slide the ribbon cable firmly and evenly into the connector until it's fully seated. Don't force it in if it encounters resistance, just ensure it's straight and properly aligned.
- Secure the Connector: Push the top part of the connector down and away from the Ethernet port, while gently holding the ribbon cable in place. This will lock the cable in place and ensure a secure connection. You should feel it click into place.
- Verify the Connection: Give the camera cable a gentle tug to ensure it's properly seated and locked. It should not come out easily.

> Mixing up the Camera (CSI) and Display (DSI) connectors is a common mistake on some Raspberry Pi boards. The connectors might look similar, but they are not interchangeable.

> The 22-pin connector has a smaller pin pitch (0.5mm) than the 15-pin connector (1mm), which is commonly used on standard Raspberry Pi models. Ensure you have the correct cable and connector for your specific Pi model and camera.

Connect USB-C Power Supply to the indicated port on the side of the Pi and plug into an outlet. Make sure to do this step last!

---

## Updating software

Once the Pi has powered on, log in to open the desktop environment

Connect to WiFi (follow the displayed steps)

Open the terminal

Run the following commands to update your system:

```bash
sudo apt update
```

This will get the latest catalog of available Linux packages

```bash
sudo apt full-upgrade
```

This will update all packages to their latest versions

---

## Enable Camera Overlay

Run:

```bash
sudo nano /boot/firmware/config,txt
```

This will open a text editor for config.txt

Find the line:

```
camera_auto_detect=1
```

and change it to:

```
camera_auto_detect=0
```

Find the line:

```
[all]
```

and add the following under it:

```
dtoverlay=imx708
```

You will need to use arrow keys to navigate the file. To save and exit, “ctrl-x” to exit the file and then “y” to indicate “yes” to save changes.

Run:

```bash
sudo reboot
```

---

# Phase 2: Video Streaming and Object Detection using YOLO AI

## Installing dependencies

Run:

```bash
sudo apt-get install -y git v4l-utils python3-picamera2
```

This will install
- git - to be able to clone git repositories
- v4l-utils - for utilities like capturing video, controlling video devices, and interacting with media hardware
- python3-picamera2 - a Python API for accessing the PiCamera

Run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This will install the UV Python package manager

Run:

```bash
sudo reboot
```

---

## Creating Python environment

Run:

```bash
uv init --app <your project name>
```

This will create a python project. Replace <your project name> with whatever you want to call your project. Generally this name should align with the goals of your project.

Run:

```bash
cd <your project name>
```

This will change your current working directory to the python project

Run:

```bash
uv venv --system-site-packages .venv/
```

This will create a virtual environment, or venv. A venv is a directory that houses a particular Python interpreter and its associated packages, creating an isolated space for your project’s dependencies. This allows you to create multiple Python project with separate venvs for different use cases and packages.

Run:

```bash
uv add picamera2
```

This will add the Python PiCamera API to your project

Run:

```bash
uv add ultralytics --optional export
```

This will add the Python ultralytics API to your project. Ultralytics is the package that allows you to implement and analyse YOLO object detection models and their performance metrics

---

### In case of the potential error “cannot import libcamera”

This means that the Linux package libcamera was not preinstalled. To fix this, you need to install the libcamera package and its dependencies:

```bash
sudo apt install libcamera-dev libcamera-apps python3-libcamera python3-kms++ libkms++-dev libfmt-dev libdrm-dev
```

---

## To install Visual Studio Code

Run:

```bash
sudo apt-get install code
```

Visual Studio Code is an IDE, or integrated development environment. Programmers need an IDE to write code, just like you need Microsoft Word or Google Docs to write essays

To open your current project in VSCode, run:

```bash
code .
```

You need to install the Python Extension to code in Python in VSCode:
- Click the extensions menu on the left side of the screen
- Search for the Python extension
- Install it

Return to your code and navigate to “main.py”

---

## Display a video stream with Python

The first step is to import the necessary libraries:

```python
from picamera2 import Picamera2
import time
import cv2
```

Picamera2 is the Python API for PiCamera that we just installed. Time is a native Python module for working with time-related operations. cv2 is a library that provides utilities for handling and processing images and video capture.

Add the python boilerplate code to run a program:

```python
def main():
    # Code goes here

if __name__ == "__main__":
    main()
```
This creates a function called `main()` and runs all the code inside (underneath and indented) the function. From now on, all code you write will go under the `main()` function.

Create and configure a `camera` object

```python
    with Picamera2() as picam2:
        config = picam2.create_video_configuration(
            # TODO Update this based on actual camera properties
            main={"size": (1280, 960), "format": "MJPEG"}
        )


       picam2.configure(config)
       picam2.start()
```

The video stream will consist of a loop that constantly processes each next image frame and displays it. To construct the loop, write this code:

```python
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Camera Stream", frame)
```
*cv2 uses the BGR color format instead of RGB, which is why we have to convert using the `cvtColor()` function.*

Lets make sure we can exit the video stream once we press Escape.

```python
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Camera Stream", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
```

*The number 27 is associated with the Escape key in ASCII*

Lets display the framerate using the `time.time()` function from the `time` module in Python

```python
        previous_frame_time = 0

        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            current_frame_time = time.time()
            dt = current_frame_time - previous_frame_time
            previous_frame_time = current_frame_time
            fps = 1.0/dt if dt > 0 else 0

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Camera Stream", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
```

Finally, once we exit the loop, we must close the camera stream window.

```python

            if cv2.waitKey(1) & 0xFF == 27:
                break
  
    cv2.destroyAllWindows()
```

Your final code should look something like this

```python
#main.py
from picamera2 import Picamera2
import cv2
import time


def main():
   with Picamera2() as picam2:
       config = picam2.create_video_configuration(
           # TODO Update this based on actual camera properties
           main={"size": (1280, 960), "format": "MJPEG"}
       )


       picam2.configure(config)
       picam2.start()


       previous_frame_time = 0


       while True:
           frame = picam2.capture_array()
           frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


           current_frame_time = time.time()
           dt = current_frame_time - previous_frame_time
           previous_frame_time = current_frame_time
           fps = 1.0/dt if dt > 0 else 0


           cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


           cv2.imshow("Camera Stream", frame)


           if cv2.waitKey(1) & 0xFF == 27:
               break
  
   cv2.destroyAllWindows()


if __name__ == "__main__":
   main()
```


The next step is to implement object detection using an NCNN YOLOv8s model.

## Open the Terminal

Install export dependencies:

```bash
uv add --dev ncnn
```

Export the YOLO model as NCNN:

```bash
yolo export model=yolov8s.pt format=ncnn
```

This will create a model file in NCNN format that can be used for efficient inference on the Raspberry Pi.

## Update our code to use the NCNN model

### Add this line to the imports section of the code

```python
#main.py
from ultralytics import YOLO
```

`YOLO` is the Ultralytics API that allows us to load and run the exported YOLOv8s NCNN model. `Picamera2`, `time`, and `cv2` remain as before.

### Next, load the model. Add this code at the top of the main function:

```python
def main():
    model = YOLO("yolov8s.pt")

```

### Next, we add code to the loop to run the object detection:

```python
    previous_frame_time = 0

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame)

        annotated_frame = results[0].plot()
```

Each frame is captured, converted from RGB to BGR for OpenCV, then passed to the YOLO model for detection. The detection results are visualized using `plot()`.

## **Your final code should look like this:**

```python
from picamera2 import Picamera2
import time
import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8s_ncnn_model")

    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        # Update this based on actual camera properties
        main={"size": (640, 480), "format": "MJPEG"}
    )

    picam2.configure(config)
    picam2.start()

    previous_frame_time = 0

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame)
        annotated_frame = results[0].plot()

        current_frame_time = time.time()
        dt = current_frame_time - previous_frame_time
        previous_frame_time = current_frame_time
        fps = 1.0/dt if dt > 0 else 0

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Stream Detections", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

# Phase 3; Optimizing YOLO Performance using HAILO Chip

The HAILO chip requires a different version of the YOLO model we've been using thus far.

## Enable PCI-E Gen3
### This boosts the speed between the HAILO Chip - a neural processing unit (NPU) - and the main central processing unit (CPU)

Open a terminal inside VSCode using Ctrl+` and run:
```bash
sudo raspi-config
```
to open the Raspberry Pi configuration menu.

- Select Advanced Options
- Select PCIe Speed
- Choose Yes to enable PCIe Gen 3 mode
- Select Finish to exit
- Select yes when asked to restart the Pi. If this is not prompted, run `sudo reboot` in the terminal

## Install the Hailo NPU dependencies:
```bash
sudo apt install -y hailo-all
```

Reboot the pi again with `sudo reboot`
To test that the AI Hat with the Hailo Chip is wired and working correctly with the Raspberry Pi, run:
```bash
hailortcli fw-control identify
```

## Download the labels for the YOLO model

The YOLO model works by classifying objects in the video stream using a pre-labeled dataset it has been trained on. This dataset is called the COCO dataset. We need to download these labels so the YOLO model knows what it's recognizing.

Run:
```bash
curl -O https://raw.githubusercontent.com/raspberrypi/picamera2/refs/heads/main/examples/hailo/coco.txt
```
Add the Hailo version of the YOLO model to the Python project:

```bash
uv add git+https://github.com/hailo-ai/hailo-apps-infra.git@2ecfc8662972ad8384546b7dbe79125dfddcb276
```

## Implementing Hailo-Accelerated YOLO Model

Start by creating a new file to write the Hailo-Accelerated model

inside the new file, put the following imports:

```python
from picamera2 import Picamera2
from picamera2.devices import Hailo
import cv2
import time
```

This time we need to add code to process each frame before feeding it to the YOLO model:

## Function: Preprocess Frame

```python
def preprocess_frame(frame, model_w, model_h):
```

Input
frame: an image (likely a NumPy array from OpenCV).

model_w, model_h: the target width and height that the frame needs to be resized to, typically required by some deep learning model.

```python
    frame_h, frame_w = frame.shape[:2]
```

extracts the height and width of the input image.

These will be used to decide how much to scale the image to fit into the model’s required input size.

```python
    scale = min(model_w / frame_w, model_h  / frame_h)
```

The idea is to resize the frame so it fits inside the target model dimensions (model_w, model_h) without changing its aspect ratio.

You compute two scale factors:

model_w / frame_w: how much you'd have to scale horizontally.

model_h / frame_h: how much you'd have to scale vertically.

Taking the min(...) ensures that after scaling, the resized image fits completely inside the target area, and nothing will get cropped.

```python
    new_w = int(frame_w * scale)
    new_h = int(frame_h * scale)
```

These are the new dimensions that keep the original aspect ratio, scaled to fit within (model_w, model_h).

```python
    resized_frame = cv2.resize(frame, (new_w, new_h))
```

Actually performs the resizing, so now we have an image of shape (new_h, new_w) that fits within the target area, but might not completely fill it.


```python
top = (model_h - new_h) // 2
bottom = model_h - new_h - top
left = (model_w - new_w) // 2
right = model_w - new_w - left
```

Since new_h and new_w are smaller than model_h and model_w, we need to add padding so the final image is exactly (model_h, model_w).

Padding is added equally to the top/bottom and left/right to center the resized image.

The subtraction and integer division ensure that if the difference is odd, the remaining pixel goes to the bottom or right.


```python
padded_frame = cv2.copyMakeBorder(
    resized_frame,
    top,
    bottom,
    left,
    right,
    borderType=cv2.BORDER_CONSTANT,
    value=(128, 128, 128),
)
```

Adds a border around the resized image so the final dimensions are exactly (model_h, model_w).

Uses a constant color (128, 128, 128) which is a middle gray in RGB (or BGR), often used as a neutral background color.

```python
return padded_frame
```

The function returns the processed frame, now of size (model_h, model_w), centered and with preserved aspect ratio, ready for input into the model.

Final function:

```python
def preprocess_frame(frame, model_w, model_h):
   frame_h, frame_w = frame.shape[:2]


   scale = min(model_w / frame_w, model_h / frame_h)
   new_w = int(frame_w * scale)
   new_h = int(frame_h * scale)


   resized_frame = cv2.resize(frame, (new_w, new_h))


   # Calculate padding amounts
   top = (model_h - new_h) // 2
   bottom = model_h - new_h - top
   left = (model_w - new_w) // 2
   right = model_w - new_w - left


   # Pad with gray (128, 128, 128)
   padded_frame = cv2.copyMakeBorder(
       resized_frame,
       top,
       bottom,
       left,
       right,
       borderType=cv2.BORDER_CONSTANT,
       value=(128, 128, 128),
   )


   return padded_frame
```

Now we need to create a function to extract the detection from the model.

## Function: `extract_detections`

**Purpose:**  
Processes raw model output (`hailo_output`) from an object detection network and converts it into a clean list of detections containing:
- class name
- bounding box in pixel coordinates
- confidence score

Only keeps detections with a confidence score above a specified threshold.

---

### Step by Step

#### Initialize results list
```python
results = []
```
Start with an empty list to collect valid detections.

---

#### Loop over classes and detections
```python
for class_id, detections in enumerate(hailo_output):
```
- `hailo_output` is a list where each element corresponds to detections for one class.
- `enumerate` gives both the class index (`class_id`) and the list of detections.

```python
    for detection in detections:
        score = detection[4]
        if score >= threshold:
```
- Each detection has at least 5 elements: first four are bounding box coordinates, the fifth is the confidence score.
- Skip detections whose score is below the given `threshold`.

---

#### Extract and scale bounding box
```python
        y0, x0, y1, x1 = detection[:4]
        bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
```
- Coordinates in `detection` are normalized between 0 and 1.
- Convert them to pixel coordinates by multiplying:
  - `x` values by image width `w`
  - `y` values by image height `h`

---

#### Append valid detection
```python
        results.append([class_names[class_id], bbox, score])
```
- Create a list containing:
  - the class name from `class_names`
  - the bounding box in pixel coordinates
  - the confidence score
- Add it to `results`.

---

#### Return all detections
```python
return results
```
- After processing all classes and detections, return the list of filtered, labeled, and scaled detections.

---

✅ **Summary:**
This function makes raw object detector output usable by:
- filtering out low-confidence detections
- converting bounding boxes to real pixel coordinates
- labeling each detection with the class name

This makes the detections ready for visualization, tracking, or further analysis.

## Function: `draw_detections`

**Purpose:**  
Draws bounding boxes and class labels with confidence scores on an image frame.

---

### Step by Step

#### Loop over each detection
```python
for class_name, bbox, score in detections:
```
- `detections` is a list of detections where each detection contains:
  - `class_name`: name of detected object class.
  - `bbox`: bounding box coordinates `(x0, y0, x1, y1)`.
  - `score`: confidence score.

---

#### Extract bounding box coordinates
```python
x0, y0, x1, y1 = bbox
```
- Unpack the bounding box pixel coordinates.

---

#### Draw rectangle on the frame
```python
cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
```
- Draw a green rectangle around the detected object.
- `(0, 255, 0)` is the color in BGR (green).
- `2` is the thickness of the rectangle border.

---

#### Create label text
```python
label = f"{class_name}: {score:.2f}"
```
- Format the label string with the class name and the confidence score rounded to 2 decimal places.

---

#### Put the label text on the frame
```python
cv2.putText(
    frame,
    label,
    (x0, y0 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    2,
)
```
- Places the label text **above** the top-left corner of the bounding box.
- Uses white color `(255, 255, 255)` for the text.
- Font size is `0.5` and thickness is `2`.

---

✅ **Summary:**  
This function visually annotates the input image with bounding boxes and text labels for all detections, making it easier to see what the model detected and how confident it is.

## Function: `main`

**Purpose:**  
Runs a real-time object detection pipeline using the Hailo model and a Raspberry Pi camera (`Picamera2`), displaying annotated detections with FPS info.

---

### Step by Step

#### Set constants
```python
MODEL_PATH = "/usr/share/hailo-models/yolov8s_h8l.hef"
LABELS_PATH = "coco.txt"
DETECTION_THRESHOLD = 0.5
```

MODEL_PATH: path to the pre-trained Hailo model.

LABELS_PATH: file containing class labels.

DETECTION_THRESHOLD: confidence threshold for filtering detections.

```python
hailo = Hailo(MODEL_PATH)
model_h, model_w = hailo.get_input_shape()[:2]
```

Create a Hailo model instance.

Get the input dimensions (height, width) expected by the model.

```python
with open(LABELS_PATH, "r") as f:
    class_names = f.read().splitlines()
```
Read the class labels into a list from the labels file.

```python
with Picamera2() as picam2:
    config = picam2.create_video_configuration({"size": (1280, 960)})
    picam2.configure(config)
    picam2.start()
```
Use Picamera2 to capture video frames at 1280x960 resolution.

```python
previous_frame_time = 0
```

Capture a frame from the camera.

Preprocess it to fit the model input size (resizing and padding).

```python
results = hailo.run(proc_frame)
detections = extract_detections(
    results, model_w, model_h, class_names, DETECTION_THRESHOLD
)
```

Run the model inference.

Extract detected objects above the threshold with bounding boxes.

```python
annotated_frame = proc_frame.copy()
draw_detections(annotated_frame, model_w, model_h, detections)
```

Make a copy of the preprocessed frame.

Draw bounding boxes and labels on the copy.


```python
current_frame_time = time.time()
dt = current_frame_time - previous_frame_time
previous_frame_time = current_frame_time
fps = 1.0 / dt if dt > 0 else 0

cv2.putText(
    annotated_frame,
    f"FPS: {fps:.2f}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)
```

Calculate frame processing speed in frames per second.

Overlay FPS text on the image.


```python
cv2.imshow("Stream Detections", annotated_frame)

if cv2.waitKey(1) & 0xFF == 27:
    break

cv2.destroyAllWindows()
```
Show video stream, exit condition, and cleanup the windows


Final Summary: This function continuously captures frames from a camera, runs an object detection model on each frame, draws detected objects and their labels, overlays the FPS, and displays the video stream in real-time. It also supports clean exit when pressing Escape.

Final code should look like this!

```python
from picamera2 import Picamera2
from picamera2.devices import Hailo
import cv2
import time


def preprocess_frame(frame, model_w, model_h):
   frame_h, frame_w = frame.shape[:2]


   scale = min(model_w / frame_w, model_h / frame_h)
   new_w = int(frame_w * scale)
   new_h = int(frame_h * scale)


   resized_frame = cv2.resize(frame, (new_w, new_h))


   # Calculate padding amounts
   top = (model_h - new_h) // 2
   bottom = model_h - new_h - top
   left = (model_w - new_w) // 2
   right = model_w - new_w - left


   # Pad with gray (128, 128, 128)
   padded_frame = cv2.copyMakeBorder(
       resized_frame,
       top,
       bottom,
       left,
       right,
       borderType=cv2.BORDER_CONSTANT,
       value=(128, 128, 128),
   )


   return padded_frame




def extract_detections(hailo_output, w, h, class_names, threshold):
   results = []
   for class_id, detections in enumerate(hailo_output):
       for detection in detections:
           score = detection[4]
           if score >= threshold:
               y0, x0, y1, x1 = detection[:4]
               bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
               results.append([class_names[class_id], bbox, score])
   return results




def draw_detections(frame, model_w, model_h, detections):
   for class_name, bbox, score in detections:
       x0, y0, x1, y1 = bbox
       cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
       label = f"{class_name}: {score:.2f}"
       cv2.putText(
           frame,
           label,
           (x0, y0 - 10),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.5,
           (255, 255, 255),
           2,
       )




def main():
   MODEL_PATH = "/usr/share/hailo-models/yolov8s_h8l.hef"
   LABELS_PATH = "coco.txt"
   DETECTION_THRESHOLD = 0.5


   hailo = Hailo(MODEL_PATH)
   model_h, model_w = hailo.get_input_shape()[:2]


   with open(LABELS_PATH, "r") as f:
       class_names = f.read().splitlines()


   with Picamera2() as picam2:
       config = picam2.create_video_configuration({"size": (1280, 960)})


       picam2.configure(config)
       picam2.start()


       previous_frame_time = 0


       while True:
           frame = picam2.capture_array()
           proc_frame = preprocess_frame(frame, model_w, model_h)


           results = hailo.run(proc_frame)
           detections = extract_detections(
               results, model_w, model_h, class_names, DETECTION_THRESHOLD
           )


           annotated_frame = proc_frame.copy()
           draw_detections(annotated_frame, model_w, model_h, detections)


           current_frame_time = time.time()
           dt = current_frame_time - previous_frame_time
           previous_frame_time = current_frame_time
           fps = 1.0 / dt if dt > 0 else 0


           cv2.putText(
               annotated_frame,
               f"FPS: {fps:.2f}",
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               1,
               (0, 255, 0),
               2,
           )


           cv2.imshow("Stream Detections", annotated_frame)


           if cv2.waitKey(1) & 0xFF == 27:
               break


   cv2.destroyAllWindows()




if __name__ == "__main__":
   main()
```

