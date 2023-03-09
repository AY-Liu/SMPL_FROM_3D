import os
import numpy as np

def get_number_dir(path):
    for root, dirs, f in os.walk(path, topdown=False):
        pass
    return len(f)

if __name__ == '__main__':
    origin_img_path = "/mnt/disk_2/aoyang/new_imgs"
    orgin_keypoints_path = "/mnt/disk_2/aoyang/new_skeleton"

    for root,_,files in os.walk(orgin_keypoints_path):
        pass

    files = sorted(files)
    for i in range(len(files)):
        name = files[i][:-4]
        kp3ds=np.load(os.path.join(orgin_keypoints_path,"{}.npy").format(name))
        img_path = os.path.join(origin_img_path,name)
        if get_number_dir(img_path) != kp3ds.shape[0]:
            print("Not matched",name,"video",get_number_dir(img_path),"skeleton",kp3ds.shape[0])