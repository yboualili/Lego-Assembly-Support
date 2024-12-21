import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Initialize camera capture from the first available camera (usually default camera)
camera = cv2.VideoCapture(0)

# Set up the device (CUDA, MPS, or CPU) based on availability
device = (
    "cuda"
    if torch.cuda.is_available()  # Use GPU if available
    else "mps"
    if torch.backends.mps.is_available()  # Use MPS (Metal Performance Shaders) on Mac if available
    else "cpu"  # Default to CPU
)

print('Using Device:', device)


def load_model():
    """
    Loads the pre-trained YOLO model.

    Returns:
        model: The loaded YOLO model.
    """
    model = YOLO('aiss_cv/yolov8/detect/yolov8s/weights/best.pt')  # Load the model from the path
    print('Model loaded successfully.')
    return model


# Load the model
model = load_model()
model.conf = 0.3  # Set the confidence threshold for object detection
CLASS_NAME_DICT = model.model.names  # Mapping of class indices to class names


def get_frame():
    """
    Captures a frame from the camera.

    Returns:
        frame (np.ndarray): The captured frame from the camera.
    """
    _, frame = camera.read()  # Read a frame from the camera
    return frame


def predict(frame):
    """
    Makes a prediction on the input frame using the YOLO model.

    Args:
        frame (np.ndarray): The input image/frame to perform object detection on.

    Returns:
        result: The detection results (bounding boxes, class ids, confidences).
    """
    result = model(frame)  # Get detection results for the frame
    return result


def plot_bboxes(frame, results):
    """
    Draws bounding boxes and labels on the frame based on the detection results.

    Args:
        frame (np.ndarray): The input image/frame where bounding boxes will be drawn.
        results: The detection results, which include bounding boxes, class ids, and confidences.

    Returns:
        frame (np.ndarray): The frame with bounding boxes and labels drawn.
    """
    # Set line width for drawing bounding boxes
    lw = max(round(sum(frame.shape) / 2 * 0.003), 2)

    # Extract bounding box coordinates, confidences, and class ids from the results
    xyxys = results[0].boxes.xyxy.numpy()  # Bounding box coordinates [xmin, ymin, xmax, ymax]
    confidences = results[0].boxes.conf.numpy()  # Detection confidences
    class_ids = results[0].boxes.cls.numpy().astype(int)  # Class ids

    # Loop through the detections and draw the bounding boxes
    for i in range(len(xyxys)):
        xyxy = xyxys[i]  # Bounding box coordinates
        pt1 = (int(xyxy[0]), int(xyxy[1]))  # Top-left corner of the bounding box
        pt2 = (int(xyxy[2]), int(xyxy[3]))  # Bottom-right corner of the bounding box

        # Generate random color for the bounding box
        color = list(np.random.random(size=3) * 256)
        text_color = (255, 255, 255) if sum(color) < 255 * 3 * 0.7 else (0, 0, 0)  # Text color

        # Get the label from the class id
        label = CLASS_NAME_DICT[class_ids[i]]

        # Draw the bounding box
        cv2.rectangle(frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

        # Draw the label background box (filled rectangle)
        tf = max(lw - 1, 1)  # Font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # Calculate text width and height
        outside = pt1[1] - h >= 3  # Check if label can fit above the box
        p2 = pt1[0] + w, pt1[1] - h - 3 if outside else pt1[1] + h + 3
        cv2.rectangle(frame, pt1, p2, color, -1, cv2.LINE_AA)  # Draw filled background for label

        # Put the class label text on the frame
        cv2.putText(frame,
                    label, (pt1[0], pt1[1] - 2 if outside else pt1[1] + h + 2),
                    0,
                    lw / 3,
                    text_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

    return frame


def draw_rectangle(frame):
    """
    Draws a static rectangle on the frame for testing purposes.

    Args:
        frame (np.ndarray): The frame to draw the rectangle on.

    Returns:
        frame (np.ndarray): The frame with the drawn rectangle.
    """
    pt1 = (100, 100)  # Top-left corner of the rectangle
    pt2 = (200, 300)  # Bottom-right corner of the rectangle

    # Generate random color for the rectangle
    color = list(np.random.random(size=3) * 256)

    # Draw the rectangle on the frame
    cv2.rectangle(frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

    return frame


def main():
    """
    The main function that continuously captures frames, performs predictions,
    and displays the results until the user presses 'q' to exit.
    """
    while True:
        frame = get_frame()  # Capture a frame
        results = predict(frame)  # Make predictions on the frame
        frame = plot_bboxes(frame, results)  # Plot bounding boxes on the frame

        # Display the frame with bounding boxes
        cv2.imshow('Bounding Boxes Test', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close all OpenCV windows


# Run the main function when the script is executed
if __name__ == '__main__':
    main()
