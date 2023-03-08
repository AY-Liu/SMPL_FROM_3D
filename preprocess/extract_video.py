import cv2
from os.path import join
import os
from tqdm import tqdm

def extract(video_path,outpath):
    video = cv2.VideoCapture(video_path)
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if os.path.exists(outpath)==False:
        os.makedirs(outpath)
    for cnt in tqdm(range(totalFrames),desc='{:10s}'.format(os.path.basename(video_path))):
            ret, frame = video.read()
            cv2.imwrite(join(outpath, '{:06d}.jpg'.format(cnt)), frame)
    video.release()

if __name__ == '__main__':
    input_path = "/mnt/disk_2/aoyang/new_video"
    output_path = "/mnt/disk_2/aoyang/new_imgs"
    for root,_,files in os.walk(input_path):
        pass
    for i in range(len(files)):
        name = files[i]
        extract(os.path.join(input_path,name),os.path.join(output_path,name))
