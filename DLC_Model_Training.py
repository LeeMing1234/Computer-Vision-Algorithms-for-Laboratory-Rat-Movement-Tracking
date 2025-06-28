import deeplabcut

# DLC folder checking and confirmation
project_folder_name = "Multiple_Rat_Detection-Multiple_Rat_Detection-2025-04-18"
video_type = "avi" #, mp4, MOV, or avi, whatever you uploaded!

# Path in Drive for video to be analysed
videofile_path = [f"C:/DeepLabCut_Data/{project_folder_name}/videos/"]
print(videofile_path)

# The prediction files and labeled videos will be saved in a output folder called `labeled-videos` folder
destfolder = f"C:/DeepLabCut_Data/{project_folder_name}/labeled-videos"

# File Direction for trained DLC model
path_config_file = f"C:/DeepLabCut_Data/{project_folder_name}/config.yaml"
print(path_config_file)

shuffle = 1 # Edit if needed; 1 is the default.

deeplabcut.create_multianimaltraining_dataset(
    path_config_file, 
    Shuffles=[shuffle],
    net_type="resnet_101",
    engine=deeplabcut.Engine.PYTORCH,
)

# Train the model using pytorch
deeplabcut.train_network(
    path_config_file,
    shuffle=shuffle,
    save_epochs=5,
    epochs=200,
    batch_size=8,
)

# Let's evaluate first:
deeplabcut.evaluate_network(path_config_file, Shuffles=[shuffle], plotting=True)

# plot a few scoremaps:
deeplabcut.extract_save_all_maps(path_config_file, shuffle=shuffle, Indices=[0, 1, 2, 3])

deeplabcut.analyze_videos(
    path_config_file,
    videofile_path,
    shuffle=shuffle,
    videotype=video_type,
    auto_track=False,
    destfolder=destfolder,
)

# Create the video with all annotation based on corresponding full.pickle file
# As reference for the user to check the detected keypoint on rat body part
deeplabcut.create_video_with_all_detections(
        path_config_file, videofile_path, shuffle=1, destfolder=destfolder,
    )

num_animals = 3 # How many animals do you expect to find?
track_type= "ellipse" # box, skeleton, ellipse(Default for this system)

# Convert the detected keypoint of rat body part to tracklets
deeplabcut.convert_detections2tracklets(
    path_config_file,
    videofile_path,
    videotype=video_type,
    shuffle=shuffle,
    track_method=track_type,
    destfolder=destfolder,
    overwrite=True,
)

# Based on the tracklets generate, stitched to form complete tracks
deeplabcut.stitch_tracklets(
    path_config_file,
    videofile_path,
    shuffle=shuffle,
    track_method=track_type,
    n_tracks=num_animals,
    destfolder=destfolder,
)

# Filter the data to remove any small jitter which may affect the overall system performance
# Generate a .h5 file and .csv file as output with keypoint position (x and y axis) and likelihood
deeplabcut.filterpredictions(
    path_config_file,
    videofile_path,
    shuffle=shuffle,
    videotype=video_type,
    track_method=track_type,
    destfolder=destfolder,
)

# Plot the trajectories to determine the p-cut off value based on likelihood
deeplabcut.plot_trajectories(
    path_config_file,
    videofile_path,
    videotype=video_type,
    shuffle=shuffle,
    track_method=track_type,
    destfolder=destfolder,
)

# Create the video with keypoint based on the filtered .csv file
# The video show the keypoint with rat (1 rat has only 1 colour for all keypoint of the rat)
deeplabcut.create_labeled_video(
    path_config_file,
    videofile_path,
    shuffle=shuffle,
    color_by="individual",
    videotype=video_type,
    save_frames=False,
    filtered=True,
    track_method=track_type,
    destfolder=destfolder,
)