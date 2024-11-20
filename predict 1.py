import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:\\Users\\Admin\\Downloads\\YGUGI\\YGUGI\\bsl_alphabet_model.h5')

# Load class labels
class_labels = ["B", "_", "C", "E", "L", "M", "O", "S", "T", "U", "W"]  # Replace with your class labels list

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to track prediction and timing
previous_class = None
start_time = 0
recognized_sequence = ""

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Define region of interest (ROI) for hand
    roi = frame[100:400, 100:400]
    
    # Preprocess the ROI for the model
    img = cv2.resize(roi, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make a prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]
    
    # Check if the predicted class is the same as the previous one
    if class_label == previous_class:
        # If the same class is predicted, check the duration
        if time.time() - start_time > 2:
            # If the class is predicted continuously for more than 2 seconds, add to recognized sequence
            recognized_sequence += class_label
            previous_class = None  # Reset to avoid duplicate entries
    else:
        # Update previous class and reset the timer
        previous_class = class_label
        start_time = time.time()
    
    # Display the prediction
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    cv2.putText(frame, class_label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the recognized sequence
    cv2.putText(frame, recognized_sequence, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame
    cv2.imshow("BSL Alphabet Recognition By LSBU CSI", frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()