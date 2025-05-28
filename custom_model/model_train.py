## Conceptual Python Script: Data Preparation for Custom Emotion Model

#This script outlines the process of extracting facial landmarks from an image dataset using MediaPipe FaceMesh and saving them in a CSV file for training an emotion classifier.

#**You will need to:**
#1.  Install necessary libraries: `opencv-python`, `mediapipe`, `numpy`.
#    ```bash
#    pip install opencv-python mediapipe numpy
#    ```
#2.  **Prepare an image dataset:**
#    * Create a root folder for your dataset (e.g., `emotion_dataset_images`).
#    * Inside this root folder, create subfolders for each emotion you want to classify (e.g., `happy`, `sad`, `angry`, `surprise`, `neutral`).
#    * Place corresponding images (JPG, PNG) into these emotion subfolders. The more diverse and numerous the images, the better your model is likely to be.
#3.  **Modify the `dataset_directory` variable in the script below to point to your dataset's root folder.**

#```python
# conceptual_data_preparation.py
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import math

# --- Initialize MediaPipe FaceMesh ---
# Using FaceMesh to get a rich set of landmarks (468 3D landmarks, or 478 if refine_landmarks=True)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,     # Process individual images
    max_num_faces=1,            # Assume one face per image for simplicity in training data
    refine_landmarks=True,      # Get more detailed landmarks including iris (478 total)
    min_detection_confidence=0.5
)

# --- Landmark Normalization Function (CRUCIAL) ---
def normalize_landmarks(image_width, image_height, landmarks_mp):
    """
    Normalizes landmarks to be relatively invariant to face scale and position.
    This is a critical step for good model performance.

    Method:
    1. Convert landmark coordinates (0-1 range) to pixel coordinates.
    2. Calculate the centroid (average position) of all landmarks.
    3. Translate all landmarks so their centroid is at the origin (0,0).
    4. Calculate a scaling factor. A common one is the mean distance of landmarks
       from the centroid, or the distance between two stable points (e.g., eye corners).
       Here, we'll use the standard deviation of landmark coordinates as a proxy for scale.
       Alternatively, one could use the bounding box of the landmarks.
    5. Scale the translated landmarks.

    Args:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        landmarks_mp (list): List of MediaPipe Landmark objects.

    Returns:
        list: A flat list of normalized x, y coordinates, or None if normalization fails.
              [norm_lm1_x, norm_lm1_y, norm_lm2_x, norm_lm2_y, ...]
    """
    if not landmarks_mp:
        return None

    pixel_landmarks = np.array([[lm.x * image_width, lm.y * image_height] for lm in landmarks_mp])

    # 1. Calculate centroid
    centroid = np.mean(pixel_landmarks, axis=0)

    # 2. Translate landmarks relative to centroid
    translated_landmarks = pixel_landmarks - centroid

    # 3. Calculate scaling factor (e.g., root mean square distance from centroid)
    # This helps make the features scale-invariant.
    rms_distance = np.sqrt(np.mean(np.sum(translated_landmarks**2, axis=1)))
    
    if rms_distance == 0: # Avoid division by zero if all points are the same (unlikely)
        return None

    # 4. Scale landmarks
    normalized_landmarks_np = translated_landmarks / rms_distance

    # Flatten the array for CSV [x0, y0, x1, y1, ...]
    # We are only using x and y for this example, but z (lm.z) could also be included.
    # If including z, ensure your normalization handles it and the feature count matches.
    flat_normalized_landmarks = normalized_landmarks_np.flatten().tolist()
    
    # Ensure correct number of features (478 landmarks * 2 coords = 956 features)
    # This number depends on refine_landmarks setting for FaceMesh
    expected_feature_count = 478 * 2 
    if len(flat_normalized_landmarks) != expected_feature_count:
        print(f"Warning: Feature count mismatch. Expected {expected_feature_count}, got {len(flat_normalized_landmarks)}")
        return None # Or handle padding/truncation if necessary and consistent with prediction

    return flat_normalized_landmarks

# --- Main Data Processing Function ---
def create_landmark_csv(dataset_path, output_csv_path):
    """
    Processes an image dataset:
    - Detects faces and extracts 478 facial landmarks using MediaPipe FaceMesh.
    - Normalizes these landmarks.
    - Saves the emotion label and normalized landmark coordinates to a CSV file.

    Args:
        dataset_path (str): Path to the root directory of the image dataset.
                            It should contain subdirectories named after emotions.
        output_csv_path (str): Path where the resulting CSV file will be saved.
    """
    print(f"Starting dataset processing from: {dataset_path}")
    
    # Define CSV header
    # First column is 'emotion', followed by landmark coordinates
    num_landmarks = 478 # Since refine_landmarks=True
    header = ['emotion']
    for i in range(num_landmarks):
        header.extend([f'lm_{i}_x', f'lm_{i}_y']) # Only x and y for this example

    # Get list of emotion subdirectories
    try:
        emotions = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if not emotions:
            print(f"Error: No emotion subdirectories found in {dataset_path}. Please check the path and dataset structure.")
            return
        print(f"Found emotion categories: {emotions}")
    except FileNotFoundError:
        print(f"Error: Dataset path {dataset_path} not found.")
        return

    processed_images_count = 0
    failed_images_count = 0

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for emotion in emotions:
            emotion_folder = os.path.join(dataset_path, emotion)
            print(f"\nProcessing emotion: '{emotion}' from folder: {emotion_folder}")
            
            image_files = [f for f in os.listdir(emotion_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                print(f"  No images found in {emotion_folder}")
                continue

            for image_name in image_files:
                image_path = os.path.join(emotion_folder, image_name)
                
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"  Warning: Could not read image {image_path}, skipping.")
                        failed_images_count +=1
                        continue
                    
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb) # Process with MediaPipe
                    
                    image_height, image_width, _ = image.shape

                    if results.multi_face_landmarks:
                        # We configured max_num_faces=1, so take the first one
                        face_landmarks_mp = results.multi_face_landmarks[0].landmark 
                        
                        # Normalize these landmarks
                        normalized_points = normalize_landmarks(image_width, image_height, face_landmarks_mp)
                        
                        if normalized_points:
                            # Create a row for the CSV: [emotion_label, lm0_x, lm0_y, lm1_x, lm1_y, ...]
                            row_data = [emotion] + normalized_points
                            writer.writerow(row_data)
                            processed_images_count += 1
                        else:
                            print(f"  Warning: Could not normalize landmarks for {image_path}, skipping.")
                            failed_images_count +=1
                    else:
                        print(f"  Warning: No face detected in {image_path}, skipping.")
                        failed_images_count +=1
                except Exception as e:
                    print(f"  Error processing image {image_path}: {e}, skipping.")
                    failed_images_count +=1
            print(f"  Finished processing for emotion '{emotion}'.")

    print(f"\n--- Data Extraction Summary ---")
    print(f"Total images processed successfully: {processed_images_count}")
    print(f"Total images failed or skipped: {failed_images_count}")
    print(f"Landmark data saved to: {output_csv_path}")
    print("Please inspect the CSV file to ensure data looks correct before training.")

# --- Main Execution ---
if __name__ == '__main__':
    # !!! IMPORTANT: SET THIS TO YOUR DATASET PATH !!!
    # Example: dataset_directory = "/Users/yourname/Desktop/emotion_dataset_images"
    dataset_directory = "./train" 
    
    csv_output_file = "facial_landmarks_for_emotion_training.csv"

    if dataset_directory is not "./train" or not os.path.exists(dataset_directory):
        print ("------------------------------data----------------------------------------",os.path.exists(dataset_directory))
        print("---------------------------------------------------------------------------")
        print("ERROR: Please update the 'dataset_directory' variable in this script")
        print("       to point to the root folder of your emotion-labeled image dataset.")
        print("       The dataset folder should contain subfolders named after emotions")
        print("       (e.g., 'happy', 'sad', 'neutral'), with images inside them.")
        print("---------------------------------------------------------------------------")
    else:
        create_landmark_csv(dataset_directory, csv_output_file)

    # Clean up MediaPipe resources
    face_mesh.close()

#Key points about this data preparation script:

#MediaPipe FaceMesh: It uses refine_landmarks=True to get 478 landmarks, which provide a detailed representation of the face, including lips, eyes, eyebrows, and face contour.

#Normalization (normalize_landmarks): This is a critical function. The example provided normalizes by translating landmarks to their centroid and then scaling by the root mean square distance from the centroid. You might need to experiment with different normalization techniques for optimal results (e.g., scaling by interocular distance, aligning based on eye and nose points). The key is that the exact same normalization must be applied during real-time prediction.

#CSV Output: It creates a CSV file where each row contains the emotion label and then 956 feature columns (478 landmarks * 2 coordinates (x, y)).

#Error Handling: Includes basic checks for missing images or undetected faces.

#Next, the conceptual model training script: