import mediapipe as mp
import cv2
import numpy as np

def draw_landmarks(image, results):
    #Draw the landmarks on the image.

    image = np.copy(image)
    # Draw landmarks for left hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    # Draw landmarks for right hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    return image

def image_process(image, model):
    #Process the image and obtain sign landmarks.
    
    # Set the image to read-only mode
    image.flags.writeable = False
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image using the model
    results = model.process(image)
    # Set the image back to writeable mode
    image.flags.writeable = True
    # Convert the image back from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    #Extract the keypoints from the sign landmarks.

    # Extract the keypoints for the left hand if present, otherwise set to zeros
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    # Extract the keypoints for the right hand if present, otherwise set to zeros
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    # Concatenate the keypoints for both hands
    keypoints = np.concatenate([lh, rh])
    return keypoints


def instruction_frame(frame, text):
    # Check if frame is valid
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Invalid frame passed to instruction_frame.")

    # Get the size of the text to display
    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    
    # Calculate text position (centered horizontally and vertically)
    text_X_coord = (frame.shape[1] - textsize[0]) // 2  # Center horizontally
    text_Y_coord = (frame.shape[0] - textsize[1]) // 2  # Center vertically

    frame_with_text = frame.copy()
    
    # Split the text into lines and draw each one
    for i, line in enumerate(text.split('\n')):  # Handle multi-line text
        y_coord = text_Y_coord + i * 30  # Adjust vertical spacing
        if y_coord + textsize[1] > frame.shape[0]:  # Check for overflow
            break  # Stop if text goes out of frame bounds
        cv2.putText(frame_with_text, line, (text_X_coord, y_coord),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame_with_text


def clear_screen(frame):
    #Clears the screen by making the frame black.
    return np.zeros_like(frame)


def check_validity(image, laplacian_threshold=40, brightness_threshold=50):
    # Check if an image is blurry or low quality.
    # Ensure image is valid
    if image is None or image.size == 0:
        return False

    # Convert to grayscale for blur detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur detection using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < laplacian_threshold:
        return False

    # Brightness check using mean pixel intensity
    mean_brightness = np.mean(gray)
    if mean_brightness < brightness_threshold:
        return False

    return True