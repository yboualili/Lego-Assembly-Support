# LEGO Assembly Support with NVIDIA Jetson Nano and CSI Camera
## Project Overview

This project aims to enhance the LEGO assembly experience by providing real-time support during the process using a web app. By leveraging an NVIDIA Jetson Nano and a CSI camera, the application detects and tracks LEGO pieces during the assembly process. The goal is to guide the user through each assembly step, ensuring they have the correct pieces and are assembling them properly. To make this process more efficient, we focused on object detection only when the camera view changes, minimizing unnecessary computations.

The web app displays the current view from the CSI camera along with the next assembly step from the LEGO manual, highlighting the required pieces visible in the camera feed. This visual aid helps the user to focus on the correct parts, streamlining the process and reducing the chance of errors.
Key Features:

- Real-time Camera Feed: Displays the current view from the camera to allow users to see the assembly in progress.
- Step-by-Step Guidance: Shows the next step from the LEGO assembly manual, ensuring users follow the correct order.
- Piece Detection: Uses object detection to identify and highlight the needed LEGO pieces in the camera view.
- Efficient Detection: Runs the object detection model only when the image changes, optimizing performance.
- Error Detection: Utilizes a Siamese model to verify whether the assembly is done correctly by comparing the current state to the expected configuration.

## Error Handling:

One of the most common assembly errors is assembling the correct pieces in the wrong way. While a typical object detection model could identify individual pieces, it would not be able to detect the configuration errors without significantly increasing the number of classes. To solve this, we trained a Siamese network to compare different assembly steps and check if the pieces were put together correctly. This added layer of verification ensures a more reliable and accurate assembly process.

This project combines computer vision, machine learning, and real-time feedback to provide a more interactive and error-resistant LEGO assembly experience.