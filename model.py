import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import pyautogui
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 1. Load the Data
def load_data(data_dir, image_size=(64, 64)):
    images = []
    labels = []
    
    # Loop through each outer folder (00, 01, 02...)
    for outer_folder in os.listdir(data_dir):
        outer_folder_path = os.path.join(data_dir, outer_folder)
        
        if os.path.isdir(outer_folder_path):
            # Now loop through the inner folders (01_palm, 02_l, etc.)
            for inner_folder in os.listdir(outer_folder_path):
                inner_folder_path = os.path.join(outer_folder_path, inner_folder)
                
                if os.path.isdir(inner_folder_path):
                    # Loop through images in the inner folder
                    for img_name in os.listdir(inner_folder_path):
                        img_path = os.path.join(inner_folder_path, img_name)
                        
                        # Load the image
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        if image is not None:
                            # Resize the image to the target size
                            image = cv2.resize(image, image_size)
                            
                            # Normalize the pixel values
                            image = image / 255.0
                            
                            # Append the image and its label
                            images.append(image)
                            labels.append(inner_folder)  # Use the inner folder name as the label

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Convert labels to numerical values
    unique_labels = {label: idx for idx, label in enumerate(np.unique(labels))}
    numerical_labels = np.array([unique_labels[label] for label in labels])

    return images, numerical_labels, unique_labels

# Path to the dataset directory
data_dir = "./dataset"

# Load the data
images, labels, unique_labels = load_data(data_dir)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape the data for the model
X_train = X_train.reshape(-1, 64, 64, 1)
X_test = X_test.reshape(-1, 64, 64, 1)

# Print shapes for verification
print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")

# 2. Build the Model
def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Adapt output layer to number of classes
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Train the Model
def train_model(model, train_images, train_labels):
    history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)
    model.save('gesture_recognition_model.h5') 
    return history  # Return the training history for graph plotting

# 4. Evaluate the Model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test Accuracy: {test_acc}')

# 5. Plot Training Statistics
def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# 6. Real-time Gesture Recognition
def recognize_gesture(model):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            # Create a blank image to draw landmarks
            landmark_image = np.zeros((64, 64, 1), dtype=np.float32)

            # Loop through the detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the blank image
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * 64)
                    y = int(landmark.y * 64)
                    cv2.circle(landmark_image, (x, y), 3, (1.0), -1)  # Use 1.0 for white color in normalized [0, 1]

            # Prepare the image for gesture recognition
            landmark_image = np.expand_dims(landmark_image, axis=0)  # Shape (1, 64, 64, 1)

            # Predict the gesture
            prediction = model.predict(landmark_image)
            command = np.argmax(prediction)

            # Map command to actions
            if command == 0:  # Left Click (Gesture "A")
                pyautogui.click()  # Simulate left mouse click
                print("Left Click")
            elif command == 1:  # Right Click (Gesture "B")
                pyautogui.click(button='right')  # Simulate right mouse click
                print("Right Click")
            elif command == 2:  # Scroll Up (Gesture "C")
                pyautogui.scroll(10)  # Scroll up
                print("Scroll Up")
            elif command == 3:  # Scroll Down (Gesture "D")
                pyautogui.scroll(-10)  # Scroll down
                print("Scroll Down")

        else:
            print("No hand detected; no action performed.")

        # Display the image with landmarks
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    # Load the data
    images, labels, unique_labels = load_data(data_dir)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Reshape the data for the model
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)

    # Build the model
    model = build_model(len(unique_labels))

    # Train the model and get training history
    history = train_model(model, X_train, y_train)

    # Plot the training accuracy and loss
    plot_training_history(history)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Recognize gestures in real time
    recognize_gesture(model)

    # Load model and recognize gestures again
    model = tf.keras.models.load_model('gesture_recognition_model.h5')
    recognize_gesture(model)
