import os
import subprocess
from tqdm import tqdm

def create_video_from_pictures(workdir,output_dir):
    # Get the directory name
    dir_name = os.path.basename(workdir)
    
    # Set the output video file name
    output_file = os.path.join(output_dir, f"{dir_name}.mp4")
    
    # Set the input file pattern and framerate
    input_files_pattern = os.path.join(workdir, "%06d.jpg")
    framerate = 30
    
    # Run the ffmpeg command to create the video
    subprocess.run(['ffmpeg', '-framerate', str(framerate), '-i', input_files_pattern, output_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE,check=True)

def create_videos_for_directories(root_dir,output_dir):
    # List all subdirectories in the root directory
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Use tqdm to display a progress bar while creating the videos
    for subdir in tqdm(subdirs, desc='Creating videos'):
        create_video_from_pictures(subdir,output_dir)

if __name__ == '__main__':
    # Example usage
    root_dir = "/mnt/disk_2/aoyang/new_imgs"
    output_dir = "/mnt/disk_2/aoyang/new_video_frame_right"
    create_videos_for_directories(root_dir,output_dir)
