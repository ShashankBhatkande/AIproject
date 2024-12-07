import numpy as np
import os
import string
import mediapipe as mp
import cv2
from functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python
import time
import random
import threading

from performance import accuracy_measurement, plot_accuracy


learning_mode_flag = False

cooldown_time = 0.5
last_time = 0
# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
model = load_model('my_model.h5')

# Create an instance of the grammar correction tool
language_tool = language_tool_python.LanguageToolPublicAPI('en-UK')

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Initial setup for the program, 
black_image = np.zeros((480, 640, 3), dtype=np.uint8)

current_frame = instruction_frame(black_image, "Press Spacebar to Clear\nPress Enter to Submit")
cv2.imshow('Camera', current_frame)
cv2.waitKey(2000)
current_frame = clear_screen(black_image)
cv2.imshow('Camera', current_frame)

current_frame = instruction_frame(black_image, "Enable Learning Mode? Press 'L'")
cv2.imshow('Camera', current_frame)
pressed_key = cv2.waitKey(0)

if pressed_key == 108:
    current_frame = instruction_frame(black_image, "Turning Learning Mode ON!")
    cv2.imshow('Camera', current_frame)
    cv2.waitKey(2000)
    learning_mode_flag = True

current_frame = instruction_frame(black_image, "Starting the Program...")
cv2.imshow('Camera', current_frame)

new_word_flag = False

# Create a thread to continuously measure accuracy.
if learning_mode_flag:
    plot_thread = threading.Thread(target=plot_accuracy, daemon=True)
    plot_thread.start()

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Run the loop while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        _, image = cap.read()
        cv2.imshow('Camera', image)

        try:
            if check_validity(image, 100, 50):
                cv2.waitKey(1000)
                continue
        
        except Exception as e:
            print("The Program is broken at the check_validity function!")

        # Process the image and obtain sign landmarks using image_process function from functions.py
        results = image_process(image, holistic)
        # Draw the sign landmarks on the image using draw_landmarks function from functions.py
        draw_landmarks(image, results)
        # Extract keypoints from the pose landmarks using keypoint_extraction function from functions.py
        keypoints.append(keypoint_extraction(results))

        # Check if 10 frames have been accumulated
        if len(keypoints) == 10:
            # Convert keypoints list to a numpy array
            keypoints = np.array(keypoints)
            # Make a prediction on the keypoints using the loaded model
            prediction = model.predict(keypoints[np.newaxis, :, :])
            confidence = np.amax(prediction)
            # Clear the keypoints list for the next set of frames
            keypoints = []

            # Check if the maximum prediction value is above 0.9
            if confidence > 0.9:
                # Check if the predicted sign is different from the previously predicted sign
                if last_prediction != actions[np.argmax(prediction)]:
                    # Append the predicted sign to the sentence list
                    sentence.append(actions[np.argmax(prediction)])
                    # Record a new prediction to use it on the next cycle
                    last_prediction = actions[np.argmax(prediction)]
                    new_word_flag = True

        # Limit the sentence length to 7 elements to make sure it fits on the screen
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset if the "Spacebar" is pressed
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

        # Check if the list is not empty
        if sentence:
            # Capitalize the first word of the sentence
            sentence[0] = sentence[0].capitalize()

        # Check if the sentence has at least two elements
        if len(sentence) >= 2:
            # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
            if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                # Check if the second last element of sentence belongs to the alphabet or is a new word
                if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    # Combine last two elements
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

        # Perform grammar check if "Enter" is pressed
        if keyboard.is_pressed('enter'):
            # Record the words in the sentence list into a single string
            text = ' '.join(sentence)
            # Apply grammar correction tool and extract the corrected result
            grammar_result = language_tool.correct(text)

        if grammar_result:
            image = np.copy(image)
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            image = cv2.putText(image, grammar_result, (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        else:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            image.setflags(write=True)
            image = cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', image)

        # If the learning mode is on and new word is detected stop the program.
        if learning_mode_flag and new_word_flag:
            # Ask the user if the prediction is correct.
            current_frame = instruction_frame(black_image, f"Is the prediction {sentence[0]} correct?")
            cv2.imshow('Camera', current_frame)

            pressed_key = cv2.waitKey(0)

            # If the user presses 'n': 
            if pressed_key == 110:
                accuracy = False
                pass
            
            # If the user presses 'y':
            elif pressed_key == 121:
                accuracy = True         # Pass the truth value to the plotting function.
                sequence= random.randint(0, 30)
                frame_path = os.path.join(PATH, sentence[0], f"{sequence}.npy")      
                keypoints = keypoint_extraction(results)
                np.save(frame_path, keypoints)          # Replace a random frame from the dataset with current frame.

            else:
                current_frame = instruction_frame(black_image, f"Press 'y' (yes) or 'n' (no)!")
                cv2.imshow('Camera', current_frame)
            
            # Reset the program.
            sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []
            new_word_flag = False

            #Plotting function
            accuracy_measurement(accuracy)

        cv2.waitKey(1)

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Shut off the server
    language_tool.close()