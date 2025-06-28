import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import mode
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from collections import defaultdict
from collections import Counter
import os
import glob
import cv2
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import time
import gc
import queue

# --- SETTINGS ---
preset_root_folder = "C:/DeepLabCut_Data"    # <<<< Set to your video folder
result_root_folder = "C:/DeepLabCut_Data"   # <<<< Set to your result folder
threshold_root_folder = "C:/DeepLabCut_Data/Movement_Threshold"   # <<<< Set to your threshold folder

# Constants
fps = 25
time_per_frame = 1 / fps

# Keypoints (used for speed & thresholding)
keypoints = [
    "Right forepaw", "Tail tip", "Tail center", "Tail base", "Left ear", "Right ear",
    "Left hind paw", "Right hind paw", "Left forepaw", "Nose", "Abdomen", "Flank",
    "Lumber", "Shoulder", "Nape", "Left eye", "Mouse", "Right eye"
]

movement_names = ["Eating", "Standing", "Inactive", "Drinking", "Walking", "Grooming", "Abnormal"]

num_keypoints = len(keypoints)
columns_per_rat = num_keypoints * 3
num_rats = 3
RAT_COLORS = {"Red": 0, "Green": 1, "Blue": 2}
CSV_OUTPUT_DIR = "C:/DeepLabCut_Data/Movement_Threshold"

# SVM model path
model_path = "C:/DeepLabCut_Data/SVM/svm_model.pkl"

# movement to be detected and it corresponding path for speed data sample
movements = {
    "Eating": "C:/DeepLabCut_Data/Movement_Threshold/Eating_Speed.csv",
    "Standing": "C:/DeepLabCut_Data/Movement_Threshold/Standing_Speed.csv",
    "Inactive": "C:/DeepLabCut_Data/Movement_Threshold/Inactive_Speed.csv",
    "Drinking": "C:/DeepLabCut_Data/Movement_Threshold/Drinking_Speed.csv",
    "Walking": "C:/DeepLabCut_Data/Movement_Threshold/Walking_Speed.csv",
    "Grooming": "C:/DeepLabCut_Data/Movement_Threshold/Grooming_Speed.csv",
    "Abnormal": "C:/DeepLabCut_Data/Movement_Threshold/Abnormal_Speed.csv",
}

# DLC folder checking and confirmation
project_folder_name = "multiple_rat-multiple_rat-2025-03-25"
video_type = "avi" #, mp4, MOV, or avi, whatever you uploaded!

# The prediction files and labeled videos will be saved in a output folder called `labeled-videos` folder
destfolder = "C:/DeepLabCut_Data/Video_Analysis/"

# File Direction for trained DLC model
path_config_file = f"C:/DeepLabCut_Data/{project_folder_name}/config.yaml"
print(path_config_file)

# --- SUB FUNCTION OF THE SYSTEM ---
def train_svm_from_files(movements_dict, keypoints, model_path):
    """
    Trains an SVM model using speed data from various movements.
    Improvements:
    - Incorporates temporal features (acceleration and windowed statistics).
    - Addresses class imbalance using `class_weight='balanced'`.
    - Performs hyperparameter tuning using GridSearchCV for C and gamma.
    """
    X, y = [], []
    
    # Store all raw dataframes to process them for temporal features
    raw_dfs_with_labels = []

    # First pass: Load data and collect raw speeds for feature engineering
    for label, (movement_name, csv_path) in enumerate(movements_dict.items()):
        try:
            df = pd.read_csv(csv_path)
            # Ensure the DataFrame has enough rows for temporal features
            if len(df) < 2: # Need at least 2 frames to calculate speed/acceleration
                print(f"[WARNING] Skipping {movement_name} due to insufficient data (less than 2 frames).")
                continue
            raw_dfs_with_labels.append((df, label, movement_name))
        except FileNotFoundError:
            print(f"[WARNING] CSV file not found for {movement_name}: {csv_path}")
            continue
        except Exception as e:
            print(f"[ERROR] Error loading CSV for {movement_name}: {e}")
            continue

    if not raw_dfs_with_labels:
        raise ValueError("No valid CSV files loaded for training!")

    # Determine expected features (number of speed columns) from the first valid DataFrame
    # This assumes all CSVs will have the same set of speed columns after processing
    first_df_speeds = [col for col in raw_dfs_with_labels[0][0].columns if 'Speed' in col and any(kp in col for kp in keypoints)]
    expected_speed_features = len(first_df_speeds)
    
    # Define window size for temporal features (e.g., 5 frames for context)
    temporal_window_size = 5 

    # Second pass: Generate rich features including temporal context
    for df, label, movement_name in raw_dfs_with_labels:
        speed_cols = [col for col in df.columns if 'Speed' in col and any(kp in col for kp in keypoints)]
        
        if len(speed_cols) != expected_speed_features:
            print(f"[WARNING] Skipping {movement_name} due to inconsistent number of speed columns. Expected {expected_speed_features}, got {len(speed_cols)}.")
            continue

        # Calculate acceleration (change in speed)
        accel_data = df[speed_cols].diff().fillna(0) # Fill NaN from diff with 0
        accel_cols = [col.replace('_Speed', '_Accel') for col in speed_cols]
        accel_data.columns = accel_cols

        # Combine speeds and accelerations
        combined_df = pd.concat([df[speed_cols], accel_data], axis=1)
        
        # Fill any remaining NaNs (e.g., from first row of diff) with 0 for feature extraction
        combined_df = combined_df.fillna(0)

        # Generate windowed features
        for i in range(len(combined_df)):
            # Ensure enough frames for the window
            if i < temporal_window_size - 1:
                continue # Skip initial frames where window cannot be fully formed

            window_start = i - temporal_window_size + 1
            window_data = combined_df.iloc[window_start : i + 1]

            # Calculate mean and standard deviation over the window for each feature
            window_features = []
            for col in combined_df.columns:
                window_features.extend([
                    window_data[col].mean(),
                    window_data[col].std() if len(window_data[col]) > 1 else 0 # Handle std for single-element window
                ])
            
            # Add current frame's raw speeds and accelerations as well
            current_frame_features = combined_df.iloc[i].values.tolist()
            
            # Combine windowed features with current frame's features
            features = current_frame_features + window_features
            
            X.append(features)
            y.append(label)

    if not X:
        raise ValueError("No valid samples found for training after feature engineering!")

    X = np.array(X)
    y = np.array(y)

    # Define the pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', probability=True, class_weight='balanced') # Added class_weight='balanced'
    )

    # Define parameters for GridSearchCV
    # These ranges can be adjusted based on initial results
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    # Use StratifiedKFold for cross-validation to maintain class balance in folds
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)

    # The best estimator from the grid search
    best_pipeline = grid_search.best_estimator_
    print(f"Best SVM parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    class_labels = list(movements_dict.keys())
    joblib.dump((best_pipeline, class_labels), model_path)
    print(f"✅ SVM model trained and saved to {model_path}")


# --- calculate_speed function (REVISED FIX) ---
def calculate_speed(df, save_path=None):
    """
    Calculates speed and acceleration for each keypoint for all rats.
    Adds acceleration data to the output DataFrame.
    REVISED: Simplified and more robust handling of NaN and diff operation for speed and acceleration.
    """
    global fps, num_rats, columns_per_rat, keypoints 

    time_per_frame = 1 / fps
    data_start_row = 3 # Assuming 3 header rows

    # Extract frame column after the header and get the number of data frames
    frame_column = df.iloc[data_start_row:, 0].reset_index(drop=True)
    all_rats_speed_data = pd.DataFrame({"Frame": frame_column})

    num_frames = len(frame_column) # The actual number of data rows

    for rat_idx in range(num_rats):
        start_col = 1 + rat_idx * columns_per_rat
        end_col = start_col + columns_per_rat
        rat_data_raw = df.iloc[data_start_row:, start_col:end_col] # Raw data for this rat

        speed_results = {}
        accel_results = {}

        for i, keypoint in enumerate(keypoints):
            x_col_idx = 3 * i
            y_col_idx = 3 * i + 1

            # Extract x and y coordinates, coercing to numeric, NaNs for non-numeric
            x_coords = pd.to_numeric(rat_data_raw.iloc[:, x_col_idx], errors='coerce').values
            y_coords = pd.to_numeric(rat_data_raw.iloc[:, y_col_idx], errors='coerce').values

            # Calculate differences for x and y. np.diff will produce NaN if input is NaN or goes from NaN to value/value to NaN
            dx = np.diff(x_coords)
            dy = np.diff(y_coords)

            # Calculate displacement. This will also have NaNs where dx or dy are NaN.
            displacement = np.sqrt(dx**2 + dy**2)
            
            # Calculate speed.
            speed = displacement / time_per_frame
            
            # Speeds are calculated between frames (e.g., speed at frame N is between frame N-1 and N).
            # The result of np.diff is one element shorter than the input.
            # So, speed has length num_frames - 1.
            # We need to prepend a NaN for the first frame (as speed for frame 0 is undefined).
            speed_full = np.insert(speed, 0, np.nan) 
            
            # Now, calculate acceleration from the `speed_full` array.
            # `np.diff` will again result in an array one element shorter.
            # Fill the first element with NaN.
            acceleration = np.diff(speed_full)
            accel_full = np.insert(acceleration, 0, np.nan)
            
            # Important: Handle cases where speed or acceleration might be NaN due to coordinates being NaN.
            # We want to keep them NaN if coordinates were missing, but if the diff operation itself introduced NaN
            # from calculation (e.g., if one coordinate was NaN and the other wasn't, resulting in NaN dx/dy),
            # we need to be careful. In this setup, if x_coords[i] or x_coords[i+1] is NaN, dx[i] will be NaN.
            # This correctly propagates the missing data.
            
            speed_results[f"Rat{rat_idx+1}_{keypoint}_Speed"] = speed_full
            accel_results[f"Rat{rat_idx+1}_{keypoint}_Accel"] = accel_full

        rat_speed_df = pd.DataFrame(speed_results)
        rat_accel_df = pd.DataFrame(accel_results)

        # Concatenate for the current rat
        all_rats_speed_data = pd.concat([all_rats_speed_data, rat_speed_df, rat_accel_df], axis=1)

    if save_path:
        # Ensure all columns are numeric, coercing errors to NaN before saving
        for col in all_rats_speed_data.columns:
            if col != "Frame":
                all_rats_speed_data[col] = pd.to_numeric(all_rats_speed_data[col], errors='coerce')
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        all_rats_speed_data.to_csv(save_path, index=False)
        print(f"📁 Speed and Acceleration data saved to: {save_path}")

    return all_rats_speed_data

# --- extract_rat_data function (No change in this fix, included for completeness) ---
def extract_rat_data(all_rats_speed_data, rat_idx):
    """
    Extracts speed and acceleration data for a specific rat from the combined DataFrame.
    """
    rat_number = rat_idx + 1
    frame_column = all_rats_speed_data["Frame"]

    # Include both Speed and Accel columns
    rat_columns = [
        col for col in all_rats_speed_data.columns
        if col.startswith(f"Rat{rat_number}_") and ('Speed' in col or 'Accel' in col)
        and any(kp in col for kp in keypoints) # Ensure it's for relevant keypoints
    ]

    rat_df = all_rats_speed_data[rat_columns].copy()
    rat_df.insert(0, "Frame", frame_column)

    return rat_df

# --- recognize_all_rats_movement_combined function (No change in this fix, included for completeness) ---
def recognize_all_rats_movement_combined(speed_df, keypoints, model_path, fps=25, plot=True, movement_path=None, recognition_path=None):
    """
    Recognizes movement for all rats using the trained SVM model.
    Improvements:
    - Ensures feature extraction matches training (speeds + accelerations + windowed stats).
    - Applies a confidence threshold for predictions.
    - Adds plt.close(fig) to prevent memory leaks from plots.
    """
    # Load model and labels
    clf, class_labels = joblib.load(model_path)

    combined_results = []
    all_recognized = []

    # Define window size for temporal features (must match training)
    temporal_window_size = 5 
    
    # Define confidence threshold for predictions
    confidence_threshold = 0.7 # Adjust this value (e.g., 0.6 to 0.9)

    for rat_id in range(1, 4):  # Loop for Rat1, Rat2, Rat3
        # Get all speed and acceleration columns for the current rat
        speed_and_accel_cols = [
            col for col in speed_df.columns 
            if f"Rat{rat_id}_" in col and ('Speed' in col or 'Accel' in col)
            and any(kp in col for kp in keypoints) # Ensure it's for relevant keypoints
        ]
        print(f"[INFO] Using {len(speed_and_accel_cols)} speed and acceleration columns for Rat {rat_id}.")

        X_pred_raw = []
        # Initialize predictions list with correct length for the current rat
        predictions = ["Unknown"] * len(speed_df)

        # Fill any NaNs with 0 before generating windowed features for prediction
        speed_df_processed = speed_df[speed_and_accel_cols].fillna(0)

        min_movement_threshold = 2 # Keep existing threshold
        max_zero_fraction = 0.95   # Keep existing threshold

        # Prepare a list to store valid frame indices for prediction
        frames_to_predict_idx = [] 

        for i in range(len(speed_df_processed)):
            # Ensure enough frames for the window
            if i < temporal_window_size - 1:
                #predictions[i] remains "Unknown" due to initialization
                continue 

            window_start = i - temporal_window_size + 1
            window_data = speed_df_processed.iloc[window_start : i + 1]

            # Calculate mean and standard deviation over the window for each feature
            window_features = []
            for col in speed_df_processed.columns:
                window_features.extend([
                    window_data[col].mean(),
                    window_data[col].std() if len(window_data[col]) > 1 else 0
                ])
            
            # Add current frame's raw speeds and accelerations as well
            current_frame_features = speed_df_processed.iloc[i].values.tolist()
            
            # Combine windowed features with current frame's features
            features = current_frame_features + window_features

            # Apply existing movement thresholding
            speeds_for_threshold = speed_df_processed.loc[i, [col for col in speed_and_accel_cols if 'Speed' in col]].values
            zero_count = np.sum(np.abs(speeds_for_threshold) < 1e-6)
            total_speed = np.sum(np.abs(speeds_for_threshold))

            if total_speed < min_movement_threshold or (zero_count / len(speeds_for_threshold)) > max_zero_fraction:
                # predictions[i] remains "Unknown" due to initialization
                continue

            X_pred_raw.append(features)
            frames_to_predict_idx.append(i) # Store the original frame index

        print(f"[INFO] Rat {rat_id} - Total frames: {len(speed_df)}, Valid frames for prediction: {len(X_pred_raw)}")

        # Make predictions and apply confidence threshold
        if X_pred_raw:
            proba = clf.predict_proba(X_pred_raw)
            preds_labels_idx = clf.predict(X_pred_raw) 

            for i, original_frame_idx in enumerate(frames_to_predict_idx):
                predicted_label_idx = preds_labels_idx[i]
                max_proba = np.max(proba[i])

                if max_proba >= confidence_threshold:
                    predictions[original_frame_idx] = class_labels[predicted_label_idx]
                else:
                    predictions[original_frame_idx] = "Unknown" 
        else:
            print(f"[WARNING] No valid frames for Rat {rat_id} after feature engineering and thresholding.")
            # predictions remains all "Unknown" due to initialization

        # Smoothing
        window = max(1, int(fps * 0.3)) 
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - window // 2)
            end = min(len(predictions), i + window // 2 + 1)
            window_vals = [predictions[j] for j in range(start, end) if predictions[j] != "Unknown"]
            
            if len(window_vals) >= window // 2: 
                smoothed.append(max(set(window_vals), key=window_vals.count))
            else:
                smoothed.append(predictions[i]) 

        rat_result = speed_df.copy()
        col_name = f"Rat{rat_id}_Predicted_Movement"
        rat_result[col_name] = smoothed
        all_recognized.append(smoothed)
        combined_results.append(rat_result[[col_name]])

        if plot:
            fig, ax = plt.subplots(figsize=(15, 4))
            plot_labels = ["Unknown"] + sorted([label for label in class_labels if label != "Unknown"])
            y_vals = [plot_labels.index(m) if m in plot_labels else 0 for m in smoothed]
            ax.plot(range(len(smoothed)), y_vals, '.-')
            ax.set_title(f"Movement Prediction for Rat {rat_id}")
            ax.set_yticks(ticks=range(len(plot_labels)), labels=plot_labels)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Predicted Movement")
            ax.grid(True)
            plt.tight_layout()

            if movement_path:
                os.makedirs(movement_path, exist_ok=True) # Ensure directory exists
                plt.savefig(os.path.join(movement_path, f"rat{rat_id}_movement_plot.png"))
                print(f"📊 Plot saved: {os.path.join(movement_path, f'rat{rat_id}_movement_plot.png')}")
            plt.close(fig) 

    prediction_df = pd.concat(combined_results, axis=1)
    final_df = pd.concat([speed_df, prediction_df], axis=1)

    # Ensure all_recognized lists are populated even if no rats found or no predictions
    # This prevents IndexError if all_recognized has fewer than 3 elements
    recognized_rat1 = all_recognized[0] if len(all_recognized) > 0 else ["Unknown"] * len(speed_df.index)
    recognized_rat2 = all_recognized[1] if len(all_recognized) > 1 else ["Unknown"] * len(speed_df.index)
    recognized_rat3 = all_recognized[2] if len(all_recognized) > 2 else ["Unknown"] * len(speed_df.index)

    summary_df = pd.DataFrame({
        "Frame": speed_df.index,
        "Rat1_Movement": recognized_rat1,
        "Rat2_Movement": recognized_rat2,
        "Rat3_Movement": recognized_rat3,
    })

    if recognition_path:
        os.makedirs(os.path.dirname(recognition_path), exist_ok=True) # Ensure directory exists
        summary_df.to_csv(recognition_path, index=False)
        print(f"✅ Saved: {recognition_path}")

    return final_df, all_recognized

# Function that use to compress the movement data from frame to second and counting of movement frequency
def compress_and_count_movements(recognized_movement, movement, fps=25, min_duration_s=3):
    # Initialize counter for each movement
    frequency = {m: 0 for m in movement}

    # Step 1: Compress movement per second
    compressed_per_sec = []
    for i in range(0, len(recognized_movement), fps):
        sec_movements = recognized_movement[i:i+fps]
        if len(sec_movements) == 0:
            continue
        # Find the most frequent movement in 1 second
        most_common = max(set(sec_movements), key=sec_movements.count)
        compressed_per_sec.append(most_common)

    # Step 2: Count only movements with at least `min_duration_s` continuous seconds
    prev_movement = None
    count = 0

    for mv in compressed_per_sec:
        if mv == prev_movement:
            count += 1
        else:
            if prev_movement in movement and count >= min_duration_s:
                frequency[prev_movement] += 1  # Movement sustained long enough
            count = 1
            prev_movement = mv

    # Handle final segment
    if prev_movement in movement and count >= min_duration_s:
        frequency[prev_movement] += 1

    return frequency

def plot_movement_histogram(movement_freq, title="Detected Movement Frequency (Compressed by Seconds)",
                            color='coral', save_path=None):
    # --- START OF MODIFICATION BLOCK ---
    fig, ax = plt.subplots(figsize=(10, 6)) # CHANGE THIS LINE: Use subplots to get a figure and axes object
    ax.bar(movement_freq.keys(), movement_freq.values(), color=color) # CHANGE THIS LINE: Plot on the 'ax' object
    ax.set_xlabel("Movement Type") # CHANGE THIS LINE: Set x-label on 'ax'
    ax.set_ylabel("Frequency (≥3s segments)") # CHANGE THIS LINE: Set y-label on 'ax'
    ax.set_title(title) # CHANGE THIS LINE: Set title on 'ax'
    plt.xticks(rotation=30)
    ax.grid(axis='y', linestyle='--', alpha=0.6) # CHANGE THIS LINE: Set grid on 'ax'
    plt.tight_layout()

    # Save to file if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"📁 Plot saved to: {save_path}")
    plt.close(fig)

def process_movement_from_csv(file_path, output_dir):
    # 1. Get filename prefix and create the new directory
    file_base = os.path.basename(file_path).split('DLC_')[0]
    save_dir = os.path.join(output_dir, file_base)
    os.makedirs(save_dir, exist_ok=True)

    # 2. Load CSV & calculate speed
    df = pd.read_csv(file_path)
    speed_data = calculate_speed(df, save_path=os.path.join(save_dir, "Speed.csv"))

    # 3. Recognised the rat movement for every frame
    recognition_path = os.path.join(save_dir, "movement_recognition.csv")

    all_rats_predictions, all_recognized = recognize_all_rats_movement_combined(
        speed_df=speed_data,
        keypoints=keypoints,
        model_path=model_path,
        fps=25,
        plot=True,
        movement_path=save_dir,
        recognition_path=recognition_path
    )

    # 4. Compress and count the rat movement
    movement_freq = {}
    total_freq = {m: 0 for m in movement_names}

    for rat_id in range(3):
        movement_freq[rat_id] = compress_and_count_movements(all_recognized[rat_id], movement_names)
        for m in movement_names:
            total_freq[m] += movement_freq[rat_id].get(m, 0)
        print(f"Rat_{rat_id+1} : {movement_freq[rat_id]}")

    # 5. Show total
    print("\nTotal Movement Frequency:")
    print(total_freq)

    # 6. Plot histogram and save
    plot_path = os.path.join(save_dir, "movement_histogram.png")
    plot_movement_histogram(total_freq, f"Total Movement Frequency", color="teal", save_path=plot_path)

def move_labeled_videos(video_path, output_path):
    # Step 1: Find all labeled videos
    video_files = glob.glob(os.path.join(video_path, "*_labeled.mp4"))

    if not video_files:
        print("❌ No labeled videos found.")
        return

    for video in video_files:
        filename = os.path.basename(video)
        
        # Step 2: Extract base name before "DLC_"
        if "DLC_" not in filename:
            print(f"⚠️ Skipping invalid filename: {filename}")
            continue
        
        base_name = filename.split("DLC_")[0]

        # Step 3: Create target folder if not exists
        target_folder = os.path.join(output_path, base_name)
        os.makedirs(target_folder, exist_ok=True)

        # Step 4: Define destination path
        dest_path = os.path.join(target_folder, filename)

        # Step 5: Move video
        shutil.move(video, dest_path)
        print(f"✅ Moved: {filename} → {target_folder}")

def extract_single_rat_speed(csv_path, rat_color, output_path, filename):
    # Map color to rat index
    color_map = {'Purple': 0, 'Red': 1, 'Green': 2}
    
    if rat_color not in color_map:
        raise ValueError("Invalid rat color. Choose from: 'Purple', 'Red', 'Green'")
    
    rat_idx = color_map[rat_color]
    rat_number = rat_idx + 1

    # Load full speed data
    df = pd.read_csv(csv_path)

    # Extract relevant columns
    frame_col = df["Frame"]
    rat_cols = [col for col in df.columns if col.startswith(f"Rat{rat_number}_")]
    rat_df = df[rat_cols].copy()

    # Strip the 'RatX_' prefix from each column name
    rat_df.columns = [col.replace(f"Rat{rat_number}_", "") for col in rat_df.columns]

    # Insert frame column at the beginning
    rat_df.insert(0, "Frame", frame_col)

    # Save to CSV
    output_file = os.path.join(output_path, f"{filename}.csv")
    rat_df.to_csv(output_file, index=False)
    print(f"✅ Saved to: {output_file}")

    return rat_df

def run_complete_dlc_analysis(
    path_config_file,
    videofile_path,
    movements,
    keypoints,
    model_path,
    destfolder=None,
    video_type='mp4',
    shuffle=1,
    num_animals=3,
    track_type="ellipse",
    output_path="C:/DeepLabCut_Data/Output"
):
    import deeplabcut

    # 1. Initial video analysis
    print("Start Analyzing my video(s)!")
    deeplabcut.analyze_videos(
        path_config_file,
        videofile_path,
        shuffle=shuffle,
        auto_track=False,
        destfolder=destfolder
    )

    # 2. Create detection reference video
    deeplabcut.create_video_with_all_detections(
        path_config_file, videofile_path, shuffle=shuffle, destfolder=destfolder
    )

    # 3. Convert to tracklets
    deeplabcut.convert_detections2tracklets(
        path_config_file,
        videofile_path,
        shuffle=shuffle,
        track_method=track_type,
        destfolder=destfolder,
        overwrite=True,
        identity_only=True,
    )

    # 4. Stitch tracklets
    deeplabcut.stitch_tracklets(
        path_config_file,
        videofile_path,
        shuffle=shuffle,
        track_method=track_type,
        n_tracks=num_animals,
        destfolder=destfolder,
        split_tracklets=False,
        min_length=5,          
        max_gap=50,
        prestitch_residuals=True
    )

    # 5. Filter predictions
    deeplabcut.filterpredictions(
        path_config_file,
        videofile_path,
        shuffle=shuffle,
        track_method=track_type,
        destfolder=destfolder,
        filtertype="median"
    )

    # 6. Plot trajectories
    deeplabcut.plot_trajectories(
        path_config_file,
        videofile_path,
        shuffle=shuffle,
        track_method=track_type,
        destfolder=destfolder,
    )

    # 7. Create final labeled video
    deeplabcut.create_labeled_video(
        path_config_file,
        videofile_path,
        shuffle=shuffle,
        color_by="individual",
        save_frames=False,
        filtered=True,
        track_method=track_type,
        destfolder=destfolder,
    )

    # 8. Process all CSV files
    csv_files = glob.glob(os.path.join(destfolder, "*.csv"))
    print(f"🔍 Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(file)
        process_movement_from_csv(file, output_path)

    # 9. Move labeled videos
    move_labeled_videos(destfolder, output_path)

# --- MAIN CLASS ---
class RatMovementApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🐀 Computer Vision for Laboratory Rat Movement Tracking")
        self.state('zoomed')  # Fullscreen but adjustable
        self.configure(bg="#e6f2ff")
        
        self.frames = {}
        for F in (WelcomePage, MainMenu, InputPage, OutputPage, SVMTrainer, Guidelines):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("WelcomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

# --- INDIVIDUAL PAGES ---
class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#e6f2ff")
        self.controller = controller

        # Main container frame for better centering
        container = tk.Frame(self, bg="#e6f2ff")
        container.pack(expand=True, fill="both", padx=20, pady=20)

        # Title label with larger font and styling
        title_label = tk.Label(
            container,
            text="Computer Vision for Laboratory Rat Movement Tracking",
            font=("Arial", 20, "bold"),
            bg="#e6f2ff",
            fg="#2c3e50",
            pady=20
        )
        title_label.pack()

        # Image display with error handling
        try:
            img = Image.open("C:/DeepLabCut_Data/Welcome_Image/Picture1.jpg")
            # Resize image if needed (example: maintain aspect ratio)
            img.thumbnail((800, 800))  # Adjust size as needed
            self.photo = ImageTk.PhotoImage(img)
            image_label = tk.Label(container, image=self.photo, bg="#e6f2ff")
            image_label.pack(pady=(0, 30))
        except Exception as e:
            print(f"Error loading image: {e}")
            # Fallback placeholder
            placeholder = tk.Label(
                container,
                text="[Welcome Image]",
                width=40,
                height=15,
                bg="white",
                relief="solid"
            )
            placeholder.pack(pady=(0, 30))

        # Enter button with improved styling
        enter_button = tk.Button(
            container,
            text="Enter",
            font=("Arial", 16),
            bg="#66b3ff",
            fg="white",
            activebackground="#4d94ff",
            activeforeground="white",
            padx=30,
            pady=10,
            relief="raised",
            borderwidth=3,
            command=lambda: controller.show_frame("MainMenu")
        )
        enter_button.pack()

        # Optional: Add some padding at the bottom
        tk.Label(container, bg="#e6f2ff").pack(pady=20)

class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#e6f2ff")
        self.controller = controller

        # Left side content
        left_frame = tk.Frame(self, bg="#e6f2ff")
        left_frame.grid(row=0, column=0, padx=50, pady=20, sticky="n")

        # Buttons
        tk.Button(left_frame, text="Input", width=20, height=2, font=("Arial", 14),
                  command=lambda: controller.show_frame("InputPage")).grid(row=0, column=0, pady=30)
        
        tk.Button(left_frame, text="Output", width=20, height=2, font=("Arial", 14),
                  command=lambda: controller.show_frame("OutputPage")).grid(row=1, column=0, pady=30)
        
        tk.Button(left_frame, text="SVM Model Update", width=20, height=2, font=("Arial", 14),
                  command=lambda: controller.show_frame("SVMTrainer")).grid(row=2, column=0, pady=30)
        
        tk.Button(left_frame, text="Guidelines", width=20, height=2, font=("Arial", 14),
                  command=lambda: controller.show_frame("Guidelines")).grid(row=3, column=0, pady=30)

        # Right side content
        right_frame = tk.Frame(self, bg="#e6f2ff")
        right_frame.grid(row=0, column=1, padx=50, pady=20, sticky="n")

        # Keypoints guideline label
        keypoints_label = tk.Label(right_frame, text="Keypoints Assign to Rat Body Part", 
                                 font=("Arial", 14, "bold"), bg="#e6f2ff")
        keypoints_label.pack(pady=(0, 10))

        # Smaller image placeholder
        try:
            # Try to load and display the image (scaled down)
            self.image = tk.PhotoImage(file="C:/DeepLabCut_Data/Welcome_Image/Welcome_Image.png")  # Replace with your image path
            # Create a smaller version of the image
            self.smaller_image = self.image.subsample(1,1)  # Reduce size by half
            image_label = tk.Label(right_frame, image=self.smaller_image, bg="#e6f2ff")
            image_label.pack()
        except:
            # Fallback if image can't be loaded (smaller placeholder)
            image_placeholder = tk.Label(right_frame, text="[Rat Keypoints Image]", 
                                       width=20, height=10, relief="solid", bg="white")
            image_placeholder.pack()

class InputPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#e6f2ff")
        self.controller = controller
        
        # Preset configuration - these are now fixed in the class
        self.base_input_path = "C:/DeepLabCut_Data/Input"
        self.output_path = "C:/DeepLabCut_Data/Output/"
        self.video_type = '.mp4'
        self.shuffle = 1
        self.num_animals = 3
        self.track_type = "ellipse"
        
        self.current_displayed_folder = None
        self.analysis_running = False
        
        # Configure grid with proper weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, uniform="col")
        self.grid_columnconfigure(1, weight=1, uniform="col")
        
        # Left Panel - Input and Controls
        left_frame = tk.Frame(self, bg="#e6f2ff")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_propagate(False)
        
        # Configure left frame grid
        left_frame.grid_rowconfigure(3, weight=1)  # Terminal area gets extra space
        left_frame.grid_columnconfigure(0, weight=1)
        
        # Back button
        back_btn = tk.Button(left_frame, text="← Back", command=lambda: controller.show_frame("MainMenu"), 
                             bg="#cccccc")
        back_btn.grid(row=0, column=0, sticky="w", pady=5)
        
        # Folder selection
        tk.Label(left_frame, text="Available Experiment Folders:", bg="#e6f2ff").grid(row=1, column=0, sticky="w")
        
        # Folder listbox with frame for better sizing
        listbox_frame = tk.Frame(left_frame, bg="#e6f2ff")
        listbox_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        listbox_frame.grid_rowconfigure(0, weight=1)
        listbox_frame.grid_columnconfigure(0, weight=1)
        
        self.folder_listbox = tk.Listbox(listbox_frame)
        self.folder_listbox.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = tk.Scrollbar(listbox_frame, command=self.folder_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.folder_listbox.config(yscrollcommand=scrollbar.set)
        
        # Status label
        self.status_label = tk.Label(left_frame, text="Select a folder from C:/DeepLabCut_Data/Input", 
                                     bg="#e6f2ff", font=("Arial", 10))
        self.status_label.grid(row=3, column=0, sticky="sw", pady=(5,0))
        
        # Terminal output with frame for better sizing
        terminal_frame = tk.Frame(left_frame, bg="#e6f2ff")
        terminal_frame.grid(row=4, column=0, sticky="nsew", pady=5)
        terminal_frame.grid_rowconfigure(0, weight=1)
        terminal_frame.grid_columnconfigure(0, weight=1)
        
        self.terminal = scrolledtext.ScrolledText(terminal_frame, state="disabled")
        self.terminal.grid(row=0, column=0, sticky="nsew")
        
        # Start analysis button
        self.analyze_btn = tk.Button(left_frame, text="Start Analysis", font=("Arial", 12), bg="#66b3ff",
                                     command=self.start_analysis)
        self.analyze_btn.grid(row=5, column=0, sticky="ew", pady=10)
        
        # Right Panel - Results Display
        right_frame = tk.Frame(self, bg="#e6f2ff")
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_propagate(False)
        
        # Configure right frame grid
        right_frame.grid_rowconfigure(1, weight=1)  # Image area gets extra space
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Results folder selection (expandable)
        self.result_selector_frame = tk.Frame(right_frame, bg="#e6f2ff")
        self.result_selector_frame.grid(row=0, column=0, sticky="ew", pady=5)
        
        tk.Label(self.result_selector_frame, text="Analysis Result:", bg="#e6f2ff").pack(side="left")
        
        # Compact view (single line)
        self.compact_view = tk.Frame(self.result_selector_frame, bg="#e6f2ff")
        self.compact_view.pack(side="left", fill="x", expand=True)
        
        self.current_result_var = tk.StringVar()
        self.result_entry = tk.Entry(self.compact_view, 
                                     textvariable=self.current_result_var,
                                     state='readonly')
        self.result_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        self.expand_btn = tk.Button(self.compact_view, text="▼", 
                                     command=self.toggle_result_list,
                                     width=3)
        self.expand_btn.pack(side="left")
        
        # Expanded view (hidden by default)
        self.expanded_view = tk.Frame(right_frame, bg="#e6f2ff")
        
        self.result_listbox = tk.Listbox(self.expanded_view)
        self.result_listbox.pack(fill="both", expand=True)
        self.result_listbox.bind("<<ListboxSelect>>", self.on_result_select)
        
        scrollbar = tk.Scrollbar(self.expanded_view, orient="vertical", command=self.result_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_listbox.config(yscrollcommand=scrollbar.set)
        
        # Image display area with consistent sizing
        self.image_frame = tk.Frame(right_frame, bg="#e6f2ff")
        self.image_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        self.image_frame.grid_propagate(False)
        
        # Configure image frame grid for a single image
        self.image_frame.grid_rowconfigure(0, weight=1) # Only one row is needed now
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        # Create only one label for the movement_recognition.png
        self.image_labels = []
        label = tk.Label(self.image_frame, bg="white", relief="solid", borderwidth=1)
        label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5) # Place in the first (and only) grid cell
        self.image_labels.append(label)
        
        # Initialize results list
        self.result_folders = []
        self.current_result_index = -1
        
        # Redirect stdout to terminal
        sys.stdout = self
        
        # Initialize folder list
        self.update_folder_list()
        self.bind("<Destroy>", self._cleanup)

    def _cleanup(self, event):
        """Clean up resources when the frame is destroyed"""
        for label in self.image_labels:
            if hasattr(label, 'image'):
                try:
                    del label.image
                except:
                    pass
        gc.collect()
    
    def write(self, text):
        """Redirect stdout to terminal widget"""
        self.terminal.config(state="normal")
        self.terminal.insert("end", text)
        self.terminal.see("end")
        self.terminal.config(state="disabled")
        # Use after() to prevent GUI lockups
        self.after(10, self.update_idletasks)
    
    def flush(self):
        pass
    
    def update_folder_list(self):
        """Update listbox with folders in base input path"""
        try:
            self.folder_listbox.delete(0, tk.END)
            
            if not os.path.exists(self.base_input_path):
                os.makedirs(self.base_input_path, exist_ok=True)
                self.folder_listbox.insert(tk.END, "No folders found")
                return
            
            folders = [f for f in os.listdir(self.base_input_path) 
                       if os.path.isdir(os.path.join(self.base_input_path, f))]
            
            if not folders:
                self.folder_listbox.insert(tk.END, "No experiment folders found")
                return
            
            for folder in sorted(folders):
                self.folder_listbox.insert(tk.END, folder)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Could not read folders: {str(e)}"))
            self.folder_listbox.insert(tk.END, "Error accessing folders")
    
    def start_analysis(self):
        """Start analysis process with thread safety checks"""
        if self.analysis_running:
            messagebox.showwarning("Warning", "Analysis is already running!")
            return
            
        selected_folder = self.folder_listbox.get(tk.ACTIVE)
        if not selected_folder or selected_folder.startswith(("No folders", "Error")):
            messagebox.showwarning("Warning", "Please select a valid folder first!")
            return
        
        # Disable button during analysis
        self.analysis_running = True
        self.analyze_btn.config(state="disabled")
        
        videofile_path = os.path.join(self.base_input_path, selected_folder)
        self.after(0, lambda: self.status_label.config(text=f"Analyzing: {selected_folder}..."))
        
        # Clear terminal
        self.terminal.config(state="normal")
        self.terminal.delete(1.0, tk.END)
        self.terminal.config(state="disabled")
        
        # Use preset parameters from the class
        params = {
            'path_config_file': path_config_file,
            'videofile_path': videofile_path,
            'keypoints': keypoints,
            'movements': movements,
            'model_path': model_path,
            'destfolder': destfolder,
            'video_type': self.video_type,
            'shuffle': self.shuffle,
            'num_animals': self.num_animals,
            'track_type': self.track_type
        }
        
        # Run analysis in thread
        analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(params, selected_folder),
            daemon=True
        )
        analysis_thread.start()
    
    def run_analysis(self, params, folder_name):
        """Run DLC analysis and update UI when complete"""
        try:
            print(f"Starting analysis of folder: {folder_name}")
            print("This may take several minutes...")
            
            run_complete_dlc_analysis(**params)
            
            # Update UI in main thread
            self.after(0, self.update_results_list)
            self.after(0, lambda: self.status_label.config(
                text=f"Analysis complete: {folder_name}"))
            self.after(0, lambda: messagebox.showinfo(
                "Complete", "Analysis finished successfully!"))
            
        except Exception as e:
            self.after(0, lambda: self.status_label.config(
                text=f"Analysis failed for {folder_name}"))
            self.after(0, lambda: messagebox.showerror(
                "Error", f"Analysis failed: {str(e)}"))
            print(f"ERROR: {str(e)}")
        finally:
            # Re-enable button
            self.after(0, lambda: setattr(self, 'analysis_running', False))
            self.after(0, lambda: self.analyze_btn.config(state="normal"))

    def toggle_result_list(self):
        """Toggle between compact and expanded views with layout stability"""
        if self.expanded_view.winfo_ismapped():
            self.expanded_view.grid_forget()
            self.expand_btn.config(text="▼")
        else:
            self.expanded_view.grid(row=2, column=0, sticky="nsew", pady=5)
            self.expand_btn.config(text="▲")
            # Ensure listbox is populated
            if not self.result_listbox.size():
                self.update_results_listbox()

    def update_results_list(self):
        """Update the list of available result folders"""
        try:
            self.result_folders = []
            if os.path.exists(self.output_path):
                self.result_folders = sorted([f for f in os.listdir(self.output_path) 
                                           if os.path.isdir(os.path.join(self.output_path, f))])
            
            self.update_results_listbox()
            
            if self.result_folders:
                self.current_result_index = 0
                self.show_result(self.result_folders[0])
            else:
                self.current_result_var.set("No results available")
                # Clear the single image label
                self.image_labels[0].config(image=None, text="No results available")
        except Exception as e:
            print(f"Error updating results list: {str(e)}")

    def update_results_listbox(self):
        """Update the listbox with current result folders"""
        self.result_listbox.delete(0, tk.END)
        for folder in self.result_folders:
            self.result_listbox.insert(tk.END, folder)

    def on_result_select(self, event):
        """Handle selection from the expanded list"""
        selection = self.result_listbox.curselection()
        if selection:
            self.current_result_index = selection[0]
            self.show_result(self.result_folders[self.current_result_index])
            # Collapse the list after selection
            self.toggle_result_list()

    def show_result(self, folder_name):
        """Display the specified result folder with memory-safe image loading"""
        try:
            # Clear current image first
            self.image_labels[0].config(image=None, text="Loading...")
            if hasattr(self.image_labels[0], 'image'):
                del self.image_labels[0].image
                self.image_labels[0].image = None
        
            # Force garbage collection
            gc.collect()
        
            self.current_result_var.set(folder_name)
            folder_path = os.path.join(self.output_path, folder_name)
            self.current_displayed_folder = folder_path
        
            # Only look for movement_recognition.png
            image_pattern = 'movement_histogram.png'
            
            # Get max width from the single visible label
            try:
                max_width = min(self.image_labels[0].winfo_width() - 20, 800) # Cap at 800px
                if max_width < 100: # Minimum reasonable width
                    max_width = 400
            except:
                max_width = 400 # Default if width can't be determined
            
            # Clear any existing image reference for the single label
            self.image_labels[0].config(image=None)
            if hasattr(self.image_labels[0], 'image'):
                del self.image_labels[0].image
                self.image_labels[0].image = None
            
            img_path = os.path.join(folder_path, image_pattern)
            
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    
                    # Calculate new dimensions to fit without cropping
                    width, height = img.size
                    if width == 0:
                        width = 1
                    
                    # Determine target size based on label's current size
                    label_width = self.image_labels[0].winfo_width()
                    label_height = self.image_labels[0].winfo_height()
                    
                    if label_width == 1 or label_height == 1: # Default values if not yet rendered
                         label_width = max_width # Use max_width as a fallback
                         label_height = 600 # Use a reasonable default height
                    
                    # Calculate aspect ratios
                    image_aspect = width / height
                    label_aspect = label_width / label_height
                    
                    new_width, new_height = width, height # Initialize with original size
                    
                    if image_aspect > label_aspect: # Image is wider than the label area
                        new_width = label_width
                        new_height = int(new_width / image_aspect)
                    else: # Image is taller than or same aspect as the label area
                        new_height = label_height
                        new_width = int(new_height * image_aspect)

                    # Ensure new dimensions are not zero or negative
                    if new_width <= 0: new_width = 1
                    if new_height <= 0: new_height = 1

                    # Verify the image is a reasonable size before processing
                    if width * height > 10000000: # 10 megapixels
                        raise MemoryError("Image too large to process safely")
                    
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(img)
                    
                    self.image_labels[0].config(
                        image=photo, 
                        text="",
                        width=new_width, # Set label size to match image size
                        height=new_height
                    )
                    self.image_labels[0].image = photo # Keep a reference
                    
                    img.close()
                    
                except MemoryError:
                    self.image_labels[0].config(
                        image=None, 
                        text=f"Image too large (Memory Error)"
                    )
                    print(f"Memory error loading {img_path}")
                except Exception as e:
                    self.image_labels[0].config(
                        image=None, 
                        text=f"Error loading image"
                    )
                    print(f"Error loading {img_path}: {str(e)}")
            else:
                self.image_labels[0].config(
                    image=None, 
                    text=f"movement_recognition.png not found"
                )
        
            # Force another garbage collection after loading
            gc.collect()
        
        except Exception as e:
            print(f"Error showing results: {str(e)}")
            self.current_result_var.set(folder_name)
            self.image_labels[0].config(image=None, text="Error displaying results")

class OutputPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(bg='white')
        
        # Set default path
        self.default_path = "C:/DeepLabCut_Data/Output"
        self.current_folder = ""
        self.current_file = ""
        
        # Video control variables
        self.video_cap = None
        self.video_playing = False
        self.video_id = None
        self.current_frame = None
        self.movement_data = None  # To store movement recognition data
        self.current_frame_num = 0  # To track current video frame
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        self.main_container = tk.Frame(self, bg='white')
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Folder and file selection (30% width)
        self.left_panel = tk.Frame(self.main_container, bg='white', width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.left_panel.pack_propagate(False)
        
        # Add Back button exactly as requested (new addition)
        tk.Button(self.left_panel, text="← Back", 
                command=lambda: self.controller.show_frame("MainMenu"),
                bg="#cccccc").pack(anchor="w", pady=5)
        
        # Right panel - Display area (70% width)
        self.right_panel = tk.Frame(self.main_container, bg='lightgray')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Folder selection
        self.folder_frame = tk.LabelFrame(self.left_panel, text="Select Folder", bg='white')
        self.folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add refresh button next to the folder label
        self.folder_header = tk.Frame(self.folder_frame, bg='white')
        self.folder_header.pack(fill=tk.X)
        tk.Label(self.folder_header, text="Folders in C:/DeepLabCut/Output", bg='white').pack(side=tk.LEFT)
        tk.Button(self.folder_header, text="↻", command=self.populate_folder_tree, 
                 bg='white', relief=tk.FLAT).pack(side=tk.RIGHT)
        
        self.folder_tree = ttk.Treeview(self.folder_frame, height=15)
        self.folder_tree.pack(fill=tk.BOTH, expand=True)
        
        self.folder_tree.heading("#0", text="", anchor=tk.W)  # Empty text since we have label above
        self.populate_folder_tree()
        
        self.folder_tree.bind("<<TreeviewSelect>>", self.on_folder_select)
        
        # File list
        self.file_frame = tk.LabelFrame(self.left_panel, text="Files in Selected Folder", bg='white')
        self.file_frame.pack(fill=tk.BOTH, expand=True)
        
        self.file_listbox = tk.Listbox(self.file_frame, selectmode=tk.SINGLE)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        
        # Display area - Now larger and more prominent
        self.display_frame = tk.Frame(self.right_panel, bg='lightgray')
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Container for video and movement display
        self.video_movement_container = tk.Frame(self.display_frame, bg='lightgray')
        self.video_movement_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for media display (larger area)
        self.media_canvas = tk.Canvas(self.video_movement_container, bg='lightgray', highlightthickness=0)
        self.media_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame for movement recognition display (horizontal layout)
        self.movement_frame = tk.Frame(self.video_movement_container, bg='white', height=50)
        self.movement_frame.pack(fill=tk.BOTH, expand=False)
        
        # Movement status labels in horizontal layout
        self.movement_label = tk.Label(self.movement_frame, text="Movement Status:", 
                                     font=('Arial', 10, 'bold'), bg='white')
        self.movement_label.pack(side=tk.LEFT, padx=5)
        
        # Create frames for each rat's movement status
        self.rat_frames = []
        self.rat_labels = []
        for i in range(3):
            frame = tk.Frame(self.movement_frame, bg='white')
            frame.pack(side=tk.LEFT, padx=10)
            self.rat_frames.append(frame)
            
            label = tk.Label(frame, text=f"Rat {i+1}: N/A", bg='white')
            label.pack()
            self.rat_labels.append(label)
        
        # Frame for CSV display (hidden by default)
        self.csv_frame = tk.Frame(self.display_frame, bg='lightgray')
        self.csv_text = tk.Text(self.csv_frame, wrap=tk.NONE, state=tk.DISABLED, bg='white')
        self.csv_scroll = tk.Scrollbar(self.csv_frame, command=self.csv_text.yview)
        self.csv_text.config(yscrollcommand=self.csv_scroll.set)
        
        self.csv_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.csv_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video control buttons
        self.video_controls = tk.Frame(self.display_frame, bg='lightgray')
        self.play_button = tk.Button(self.video_controls, text="▶", command=self.play_video)
        self.pause_button = tk.Button(self.video_controls, text="⏸", command=self.pause_video)
        self.stop_button = tk.Button(self.video_controls, text="⏹", command=self.stop_video)
        
        self.play_button.pack(side=tk.LEFT, padx=5)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Bind canvas resize
        self.media_canvas.bind("<Configure>", self.on_canvas_resize)
        
    def populate_folder_tree(self):
        """Populate the folder tree with subdirectories of the default path"""
        self.folder_tree.delete(*self.folder_tree.get_children())
        
        if not os.path.exists(self.default_path):
            messagebox.showerror("Error", f"Default path {self.default_path} does not exist!")
            return
            
        for folder in os.listdir(self.default_path):
            folder_path = os.path.join(self.default_path, folder)
            if os.path.isdir(folder_path):
                self.folder_tree.insert("", tk.END, text=folder, values=[folder_path])
    
    def on_folder_select(self, event):
        """Handle folder selection"""
        selected_item = self.folder_tree.selection()
        if not selected_item:
            return
            
        folder_name = self.folder_tree.item(selected_item, "text")
        self.current_folder = os.path.join(self.default_path, folder_name)
        
        # Populate file listbox
        self.file_listbox.delete(0, tk.END)
        
        if not os.path.exists(self.current_folder):
            messagebox.showerror("Error", "Selected folder does not exist!")
            return
            
        for file in os.listdir(self.current_folder):
            if file.lower().endswith(('.png', '.csv', '.mp4')):
                self.file_listbox.insert(tk.END, file)
    
    def on_file_select(self, event):
        """Handle file selection and display content"""
        selected_index = self.file_listbox.curselection()
        if not selected_index:
            return
            
        self.current_file = self.file_listbox.get(selected_index)
        file_path = os.path.join(self.current_folder, self.current_file)
        
        # Clear previous display
        self.clear_display()
        
        if self.current_file.lower().endswith('.png'):
            self.display_image(file_path)
        elif self.current_file.lower().endswith('.mp4'):
            self.display_video(file_path)
        elif self.current_file.lower().endswith('.csv'):
            self.display_csv(file_path)
    
    def clear_display(self):
        """Clear the display area"""
        # Stop any playing video
        self.stop_video()
        
        # Hide CSV frame and show media canvas
        self.csv_frame.pack_forget()
        self.media_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Clear canvas
        self.media_canvas.delete("all")
        
        # Clear CSV text
        self.csv_text.config(state=tk.NORMAL)
        self.csv_text.delete(1.0, tk.END)
        self.csv_text.config(state=tk.DISABLED)
        
        # Hide video controls
        self.video_controls.pack_forget()
        
        # Hide movement frame
        self.movement_frame.pack_forget()
    
    def display_image(self, image_path):
        """Display PNG image centered and scaled to fit canvas"""
        try:
            # Hide video controls
            self.video_controls.pack_forget()
            
            # Load image
            img = Image.open(image_path)
            
            # Calculate scaling to fit canvas while maintaining aspect ratio
            canvas_width = self.media_canvas.winfo_width()
            canvas_height = self.media_canvas.winfo_height()
            
            img_ratio = img.width / img.height
            canvas_ratio = canvas_width / canvas_height
            
            if img_ratio > canvas_ratio:
                # Image is wider than canvas
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                # Image is taller than canvas
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            
            # Calculate position to center image
            x_pos = (canvas_width - new_width) // 2
            y_pos = (canvas_height - new_height) // 2
            
            # Display image on canvas
            self.media_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_img)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def load_movement_data(self, video_path):
        """Load movement recognition data for the selected video"""
        # Find the corresponding movement_recognition.csv file
        folder = os.path.dirname(video_path)
        csv_file = os.path.join(folder, "movement_recognition.csv")
        
        if not os.path.exists(csv_file):
            # Hide movement frame if no data found
            self.movement_frame.pack_forget()
            self.movement_data = None
            return
            
        try:
            self.movement_data = pd.read_csv(csv_file)
            # Show movement frame
            self.movement_frame.pack(fill=tk.BOTH, expand=False)
            self.update_movement_text(0)  # Initialize text at frame 0
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not load movement data: {str(e)}")
            self.movement_frame.pack_forget()
            self.movement_data = None
    
    def update_movement_text(self, frame_num):
        """Update the movement text to show current frame status"""
        if self.movement_data is None:
            return
            
        # Get movement data for current frame
        frame_data = self.movement_data[self.movement_data['Frame'] == frame_num]
        
        # Update each rat's label
        for i in range(3):
            rat_col = f'Rat{i+1}_Movement'
            if rat_col in frame_data.columns:
                self.rat_labels[i].config(text=f"Rat {i+1}: {frame_data[rat_col].values[0]}")
            else:
                self.rat_labels[i].config(text=f"Rat {i+1}: N/A")
    
    def display_video(self, video_path):
        """Setup video display and start playback"""
        try:
            # Show video controls
            self.video_controls.pack(fill=tk.X, pady=(0, 8))
            
            # Load corresponding movement data
            self.load_movement_data(video_path)
            
            # Open video file
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video FPS to determine playback speed
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.frame_delay = int(1000 / fps) if fps > 0 else 33  # Default to ~30fps if fps is 0
            
            # Reset frame counter
            self.current_frame_num = 0
            
            # Start playback automatically
            self.play_video()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display video: {str(e)}")
    
    def play_video(self):
        """Start video playback"""
        if self.video_cap is not None and not self.video_playing:
            self.video_playing = True
            self.update_video()
    
    def pause_video(self):
        """Pause video playback"""
        self.video_playing = False
        if self.video_id:
            self.after_cancel(self.video_id)
            self.video_id = None
    
    def stop_video(self):
        """Stop video and return to first frame"""
        self.pause_video()
        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_num = 0
            if self.movement_data is not None:
                self.update_movement_text(0)
            self.update_video()
    
    def update_video(self):
        """Update video frame - improved for smoother playback"""
        if self.video_playing and self.video_cap is not None:
            start_time = time.time()  # For performance measurement
            
            ret, frame = self.video_cap.read()
            
            if ret:
                # Get current frame number
                self.current_frame_num = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # Update movement text
                if self.movement_data is not None:
                    self.update_movement_text(self.current_frame_num)
                
                # Convert frame to RGB and then to ImageTk format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                self.current_frame = img  # Store current frame for resize handling
                
                # Calculate scaling to fit canvas while maintaining aspect ratio
                canvas_width = self.media_canvas.winfo_width()
                canvas_height = self.media_canvas.winfo_height()
                
                img_ratio = img.width / img.height
                canvas_ratio = canvas_width / canvas_height
                
                if img_ratio > canvas_ratio:
                    # Video is wider than canvas
                    new_width = canvas_width
                    new_height = int(canvas_width / img_ratio)
                else:
                    # Video is taller than canvas
                    new_height = canvas_height
                    new_width = int(canvas_height * img_ratio)
                
                # Resize image
                img = img.resize((new_width, new_height), Image.LANCZOS)
                self.tk_img = ImageTk.PhotoImage(image=img)
                
                # Calculate position to center video
                x_pos = (canvas_width - new_width) // 2
                y_pos = (canvas_height - new_height) // 2
                
                # Display frame on canvas
                self.media_canvas.delete("all")
                self.media_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_img)
                
                # Calculate processing time and adjust delay
                processing_time = (time.time() - start_time) * 1000  # in ms
                adjusted_delay = max(1, self.frame_delay - int(processing_time))
                
                # Schedule next frame update
                self.video_id = self.after(adjusted_delay, self.update_video)
            else:
                # End of video - restart from beginning
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_num = 0
                self.update_video()
    
    def display_video_frame(self, img):
        """Display a single video frame (for resize handling)"""
        if img is None:
            return
            
        # Calculate scaling to fit canvas while maintaining aspect ratio
        canvas_width = self.media_canvas.winfo_width()
        canvas_height = self.media_canvas.winfo_height()
        
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            # Video is wider than canvas
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            # Video is taller than canvas
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        # Resize image
        img = img.resize((new_width, new_height), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(image=img)
        
        # Calculate position to center video
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        
        # Display frame on canvas
        self.media_canvas.delete("all")
        self.media_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_img)
    
    def display_csv(self, csv_path):
        """Display CSV data with scrollable grid"""
        try:
            # Hide media canvas and show CSV frame
            self.media_canvas.pack_forget()
            self.csv_frame.pack(fill=tk.BOTH, expand=True)
            
            # Hide video controls
            self.video_controls.pack_forget()
            
            df = pd.read_csv(csv_path)
            
            # Create a string representation of the dataframe
            csv_content = df.to_string(index=False)
            
            # Configure text widget
            self.csv_text.config(state=tk.NORMAL)
            self.csv_text.delete(1.0, tk.END)
            self.csv_text.insert(tk.END, csv_content)
            self.csv_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display CSV: {str(e)}")
    
    def on_canvas_resize(self, event):
        """Handle canvas resize events"""
        if hasattr(self, 'current_file') and self.current_file:
            if self.current_file.lower().endswith('.png') and hasattr(self, 'tk_img'):
                self.display_image(os.path.join(self.current_folder, self.current_file))
            elif self.current_file.lower().endswith('.mp4') and self.current_frame is not None:
                self.display_video_frame(self.current_frame)

class SVMTrainer(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#e6f2ff")
        self.controller = controller
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Initialize paths and variables
        self.output_dir = "C:/DeepLabCut_Data/Output"
        self.movement_threshold_dir = "C:/DeepLabCut_Data/Movement_Threshold"
        self.model_path = "C:/DeepLabCut_Data/SVM/svm_model.pkl"
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.movement_threshold_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.video_path = ""
        self.cap = None
        self.playing = False
        self.current_frame = None
        self.current_csv_path = ""
        self.video_thread = None
        self.video_lock = threading.Lock()
        
        # Initialize movements dictionary
        self.movements = {
            "Eating": os.path.join(self.movement_threshold_dir, "Eating_Speed.csv"),
            "Standing": os.path.join(self.movement_threshold_dir, "Standing_Speed.csv"),
            "Inactive": os.path.join(self.movement_threshold_dir, "Inactive_Speed.csv"),
            "Drinking": os.path.join(self.movement_threshold_dir, "Drinking_Speed.csv"),
            "Walking": os.path.join(self.movement_threshold_dir, "Walking_Speed.csv"),
            "Grooming": os.path.join(self.movement_threshold_dir, "Grooming_Speed.csv"),
            "Abnormal": os.path.join(self.movement_threshold_dir, "Abnormal_Speed.csv"),
        }

        # GUI variables
        self.color_var = tk.StringVar(value="Purple")
        self.filename_var = tk.StringVar()
        self.status_var = tk.StringVar(value="🕘 Status: Waiting for user input...")
        self.retrain_var = tk.StringVar(value="")

        # Build the interface
        self.initialize_widgets()
        self.update_folder_list() # Initial population of the folder list

    def initialize_widgets(self):
        """Initialize all widgets without binding methods that don't exist yet"""
        # Back button
        tk.Button(self, text="← Back", command=lambda: self.controller.show_frame("MainMenu"),
                  bg="#cccccc", font=("Arial", 10)).place(x=10, y=10)

        # Folder selection label
        folder_label = tk.Label(self, text="📁 Select Video Folder:", bg="#e6f2ff", font=("Arial", 12))
        folder_label.place(x=20, y=60)

        # Refresh button for folder list
        refresh_button = tk.Button(self, text="🔄 Refresh", command=self.update_folder_list,
                                   bg="#aaddff", font=("Arial", 10))
        # Place it next to the folder label
        refresh_button.place(x=folder_label.winfo_reqwidth() + 30, y=60) # Adjust x based on label width and padding

        self.folder_listbox = tk.Listbox(self, width=50, height=15)
        self.folder_listbox.place(x=20, y=90)

        # Movement name input
        tk.Label(self, text="🎯 Movement Name:", bg="#e6f2ff", font=("Arial", 12)).place(x=400, y=90)
        self.movement_combobox = ttk.Combobox(self, textvariable=self.filename_var, 
                                               values=list(self.movements.keys()),
                                               state="readonly", width=27, font=("Arial", 12))
        self.movement_combobox.place(x=400, y=120)

        # Rat color selection
        tk.Label(self, text="🎨 Rat Color:", bg="#e6f2ff", font=("Arial", 12)).place(x=400, y=160)
        ttk.Combobox(self, textvariable=self.color_var, values=["Purple", "Red", "Green"],
                     state="readonly", width=27, font=("Arial", 12)).place(x=400, y=190)

        # Extract speed button
        tk.Button(self, text="📤 Extract Speed", bg="#66b3ff", font=("Arial", 12),
                  command=self.extract_speed).place(x=400, y=240)

        # Current movements display
        tk.Label(self, text="🧠 Current Movement Files:", bg="#e6f2ff", font=("Arial", 12)).place(x=400, y=290)
        self.movement_text = tk.Text(self, width=40, height=8, font=("Arial", 10))
        self.movement_text.place(x=400, y=320)

        # Video preview
        self.video_label = tk.Label(self, text="🎞 Video Preview", width=600, height=600, bg="#d9d9d9")
        self.video_label.place(x=750, y=90)

        # Retrain options
        self.retrain_frame = tk.Frame(self, bg="#e6f2ff")
        self.retrain_frame.place(x=400, y=520)
        tk.Label(self.retrain_frame, text="🔁 Retrain model with new data?", bg="#e6f2ff", 
                 font=("Arial", 12)).pack(side="left")
        tk.Button(self.retrain_frame, text="Yes", command=lambda: self.retrain_model(True), 
                  bg="#66ff66", font=("Arial", 12)).pack(side="left", padx=5)
        tk.Button(self.retrain_frame, text="No", command=lambda: self.retrain_model(False), 
                  bg="#ff6666", font=("Arial", 12)).pack(side="left", padx=5)
        self.retrain_frame.place_forget()

        # Status bar
        self.status_label = tk.Label(self, textvariable=self.status_var, anchor="w",
                                     bg="#e6f2ff", font=("Arial", 10), relief="sunken")
        self.status_label.place(x=10, y=730, width=1350)

        # Now that all methods exist, set up bindings
        self.setup_bindings()
        self.update_movement_display()

    def setup_bindings(self):
        """Set up all widget bindings after methods are defined"""
        self.folder_listbox.bind("<<ListboxSelect>>", self.load_video_preview)
        self.movement_combobox.bind("<<ComboboxSelected>>", self.update_filename_from_combobox)

    def update_folder_list(self):
        """Update the folder listbox with available output folders"""
        self.folder_listbox.delete(0, tk.END)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            messagebox.showinfo("Info", f"Output directory created: {self.output_dir}")
            return
        
        folders = [f for f in os.listdir(self.output_dir)
                   if os.path.isdir(os.path.join(self.output_dir, f))]
        
        if not folders:
            self.folder_listbox.insert(tk.END, "No video folders found")
            self.status_var.set("ℹ️ No video folders found in output directory.")
            return

        for f in sorted(folders): # Sort for consistent display
            self.folder_listbox.insert(tk.END, f)
        self.status_var.set("✅ Folder list refreshed.")


    def update_filename_from_combobox(self, event=None):
        """Update filename when movement is selected from combobox"""
        selected_movement = self.movement_combobox.get()
        self.filename_var.set(selected_movement)

    def load_video_preview(self, event=None):
        """Load and display the selected video"""
        selected = self.folder_listbox.get(tk.ACTIVE)
        if not selected or selected == "No video folders found":
            self.status_var.set("⚠️ Please select a valid folder.")
            return
            
        folder_path = os.path.join(self.output_dir, selected)
        
        # Find the labeled video and corresponding CSV
        video_files = [f for f in os.listdir(folder_path) if f.endswith("_labeled.mp4")]
        csv_files = [f for f in os.listdir(folder_path) if f.endswith("Speed.csv")]
        
        if not video_files or not csv_files:
            self.status_var.set("❌ No labeled video or speed CSV found in selected folder.")
            self.stop_video() # Ensure video is stopped if no files are found
            self.video_label.config(image=None, text="🎞 Video Preview") # Clear preview
            return
            
        self.current_csv_path = os.path.join(folder_path, csv_files[0])
        self.video_path = os.path.join(folder_path, video_files[0])
        
        # Start video playback
        self.play_video(self.video_path)
        self.status_var.set(f"✅ Loaded: {selected}")

    def safe_video_capture(self, path):
        """Safely create a new video capture object"""
        with self.video_lock:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            return self.cap.isOpened()

    def play_video(self, path):
        """Start playing the video at the given path"""
        self.stop_video()
        
        if not self.safe_video_capture(path):
            self.status_var.set("❌ Failed to open video.")
            return

        self.playing = True
        self.video_thread = threading.Thread(target=self._update_video_frame, daemon=True)
        self.video_thread.start()

    def stop_video(self):
        """Safely stop video playback"""
        self.playing = False
        if self.video_thread and self.video_thread.is_alive():
            # Give the thread a moment to finish its current loop iteration
            self.video_thread.join(timeout=0.1) 
        
        with self.video_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        self.video_label.config(image=None) # Clear the image on the label
        self.current_frame = None # Clear the PhotoImage reference

    def _update_video_frame(self):
        """Thread function to update video frames"""
        try:
            while self.playing:
                with self.video_lock:
                    if not self.cap or not self.cap.isOpened():
                        break
                        
                    ret, frame = self.cap.read()
                    if not ret:
                        # Loop video if it ends
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                        
                # Process frame outside the lock
                # Resize frame to fit the label, maintaining aspect ratio
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()

                if label_width == 1 or label_height == 1: # Fallback if label size not yet determined
                    label_width = 600
                    label_height = 600

                h, w, _ = frame.shape
                aspect_ratio = w / h

                if aspect_ratio > (label_width / label_height): # Video is wider than label
                    new_width = label_width
                    new_height = int(new_width / aspect_ratio)
                else: # Video is taller than or same aspect as label
                    new_height = label_height
                    new_width = int(new_height * aspect_ratio)

                # Ensure dimensions are positive
                if new_width <= 0: new_width = 1
                if new_height <= 0: new_height = 1

                frame = cv2.resize(frame, (new_width, new_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Update current frame
                self.current_frame = ImageTk.PhotoImage(Image.fromarray(frame))
                
                # Schedule GUI update
                self.after(0, self._update_image)
                time.sleep(0.03)    # ~30fps
                
        except Exception as e:
            print(f"Video playback error: {e}")
            self.after(0, lambda: self.status_var.set(f"Video error: {str(e)}"))
        finally:
            self.playing = False # Ensure playing is set to False on exit
            self.after(0, lambda: self.video_label.config(image=None, text="🎞 Video Preview")) # Clear image on error/exit


    def _update_image(self):
        """Update image in main thread"""
        if self.playing and self.current_frame:
            self.video_label.configure(image=self.current_frame)
            self.video_label.image = self.current_frame

    def update_movement_display(self):
        """Update the movement text display with current movements"""
        self.movement_text.config(state="normal") # Enable editing
        self.movement_text.delete(1.0, tk.END)
        for movement, path in self.movements.items():
            file_exists = os.path.exists(path)
            status = "✅" if file_exists else "❌"
            self.movement_text.insert(tk.END, f"{status} {movement}: {path}\n")
            
        # Color code existing/missing files
        self.movement_text.tag_configure("exists", foreground="green")
        self.movement_text.tag_configure("missing", foreground="red")
        
        for i, (movement, path) in enumerate(self.movements.items()):
            if os.path.exists(path):
                self.movement_text.tag_add("exists", f"{i+1}.0", f"{i+1}.end")
            else:
                self.movement_text.tag_add("missing", f"{i+1}.0", f"{i+1}.end")
        self.movement_text.config(state="disabled") # Disable editing

    def extract_speed(self):
        """Extract speed data with threading"""
        def _extract():
            try:
                movement_name = self.filename_var.get().strip()
                if not movement_name:
                    self.after(0, lambda: messagebox.showwarning("Input Error", "Please select a movement name."))
                    raise ValueError("Please select a movement name.")
                    
                rat_color = self.color_var.get()
                if not self.current_csv_path:
                    self.after(0, lambda: messagebox.showwarning("Input Error", "Please select a folder with speed data."))
                    raise ValueError("Please select a folder with speed data.")
                    
                output_path = os.path.join(self.movement_threshold_dir, f"{movement_name}_Speed.csv")
                
                self.after(0, lambda: self.status_var.set("⏳ Extracting speed data..."))
                
                # This should be your actual extraction function
                extract_single_rat_speed(
                    csv_path=self.current_csv_path,
                    rat_color=rat_color,
                    output_path=self.movement_threshold_dir,
                    filename=f"{movement_name}_Speed"
                )
                
                self.after(0, lambda: self._on_extraction_success(movement_name, output_path))
                
            except Exception as e:
                self.after(0, lambda: self.status_var.set(f"❌ Error: {str(e)}"))
                print(f"Extraction error: {e}")

        self.executor.submit(_extract)

    def _on_extraction_success(self, movement_name, output_path):
        """Handle successful extraction in main thread"""
        self.movements[movement_name] = output_path
        self.update_movement_display()
        self.status_var.set(f"✅ Successfully extracted {movement_name} speed data.")
        self.retrain_frame.place(x=400, y=520)
        
    def retrain_model(self, do_retrain):
        """Improved model training with threading"""
        if not do_retrain:
            self.status_var.set("ℹ️ Model retraining skipped.")
            self.retrain_frame.place_forget()
            return
            
        def _train():
            try:
                self.after(0, lambda: self.status_var.set("⏳ Training SVM model..."))
                
                keypoints = [
                    "Right forepaw", "Tail tip", "Tail center", "Tail base", 
                    "Left ear", "Right ear", "Left hind paw", "Right hind paw", 
                    "Left forepaw", "Nose", "Abdomen", "Flank", "Lumber", 
                    "Shoulder", "Nape", "Left eye", "Mouse", "Right eye"
                ]
                
                # This should be your actual training function
                train_svm_from_files(
                    movements_dict=self.movements,
                    keypoints=keypoints,
                    model_path=self.model_path
                )
                
                self.after(0, self._on_training_success)
                
            except Exception as e:
                self.after(0, lambda: self.status_var.set(f"❌ Training failed: {str(e)}"))
                print(f"Training error: {e}")

        self.executor.submit(_train)

    def _on_training_success(self):
        """Handle successful training in main thread"""
        self.status_var.set("✅ Model retrained successfully!")
        self.retrain_frame.place_forget()
        
    def on_close(self):
        """Clean up resources when closing"""
        self.stop_video()
        self.executor.shutdown(wait=False)
        # self.cap is released in stop_video, no need to release again here
        print("SVMTrainer resources cleaned up.")

class Guidelines(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#e6f2ff")
        self.controller = controller
        
        # Main container
        container = tk.Frame(self, bg="#e6f2ff")
        container.pack(expand=True, padx=50, pady=50)
        
        # Title
        title_label = tk.Label(
            container,
            text="Directory Setup Guidelines",
            font=("Arial", 16, "bold"),
            bg="#e6f2ff",
            fg="#2c3e50"
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Table headers
        header1 = tk.Label(
            container,
            text="Directory Type",
            font=("Arial", 12, "bold"),
            bg="#e6f2ff",
            fg="#2c3e50",
            padx=10,
            pady=5
        )
        header1.grid(row=1, column=0, sticky="w")
        
        header2 = tk.Label(
            container,
            text="Path",
            font=("Arial", 12, "bold"),
            bg="#e6f2ff",
            fg="#2c3e50",
            padx=10,
            pady=5
        )
        header2.grid(row=1, column=1, sticky="w")
        
        # Directory data
        directories = [
            ("Input Video:", "C:/DeepLabCut_Data/Input"),
            ("Destination Folder (DLC):", "C:/DeepLabCut_Data/Video_Analysis"),
            ("Output Path:", "C:/DeepLabCut_Data/Output"),
            ("SVM Model Path:", "C:/DeepLabCut_Data/SVM"),
            ("DLC model path:", "C:/DeepLabCut_Data/Your_Project_Name")
        ]
        
        # Create table rows
        for i, (dir_type, path) in enumerate(directories, start=2):
            # Directory type label
            type_label = tk.Label(
                container,
                text=dir_type,
                font=("Arial", 11),
                bg="#e6f2ff",
                padx=10,
                pady=5,
                anchor="w"
            )
            type_label.grid(row=i, column=0, sticky="w")
            
            # Path label
            path_label = tk.Label(
                container,
                text=path,
                font=("Arial", 11, "italic"),
                bg="#e6f2ff",
                fg="#0066cc",
                padx=10,
                pady=5,
                anchor="w"
            )
            path_label.grid(row=i, column=1, sticky="w")
        
        # Back button
        back_button = tk.Button(
            container,
            text="Back to Main Menu",
            font=("Arial", 12),
            bg="#66b3ff",
            fg="white",
            command=lambda: controller.show_frame("MainMenu"),
            padx=15,
            pady=5
        )
        back_button.grid(row=len(directories)+2, column=0, columnspan=2, pady=(30, 0))


# --- RUN APP ---
if __name__ == "__main__":
    app = RatMovementApp()
    app.mainloop()