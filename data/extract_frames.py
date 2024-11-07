import cv2
import os

def extract_frames(video_path, output_dir, fps=1):
    # Get the base name of the video file (e.g., Drones_1 or Drones_2)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create a subdirectory within output_dir for each video
    video_frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_frame_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    count = 0
    frame_rate = int(video.get(cv2.CAP_PROP_FPS) // fps)  # Calculate the interval for frame extraction

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frame = cv2.resize(frame, (256, 256))
            frame_name = f"frame_{count}.jpg"
            cv2.imwrite(os.path.join(video_frame_dir, frame_name), frame)
        count += 1

    video.release()

# Example usage:
extract_frames("images/mp4/Drones_1.mp4", "images/frames")
extract_frames("images/mp4/Drones_2.mp4", "images/frames")
