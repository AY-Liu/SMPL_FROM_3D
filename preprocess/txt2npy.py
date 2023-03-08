import numpy as np
from tqdm import tqdm
import os

def get_3d_skeleton(path):
    f = open(path,"r",encoding='UTF-8')
    data=f.readlines()
    save_index=[]
    for i in range(3,len(data),90):
        save_index.append(i)
    new_data=[data[i:i+88] for i in save_index]
    new_data=np.array(new_data)
    points_id=[1 , 2 , 3 ,4 ,5, 18 ,7 , 9 ,23 ,49,24 ,50 ,25, 51, 43, 69 ,74 ,81, 75 ,82, 76 ,83 ,77, 84]
    nrow=len(new_data)
    final_data=[None]*nrow
    for i in range(nrow):
        final_data[i]=list(new_data[i][points_id])
    ncol=len(final_data[0])
    for i in range(nrow):
        for j in range(ncol):
            final_data[i][j]=final_data[i][j].split(" ")[0:3]
    final_data=np.array(final_data).astype(np.float32)
    return final_data

if __name__ == '__main__':
    origin_skeleton_path="/mnt/disk_2/aoyang/new_txt"
    save_keypoints_path="/mnt/disk_2/aoyang/new_skeleton"
    for root, _, names in os.walk(origin_skeleton_path):
        break
    for i in tqdm(range(len(names))):
        name=names[i][:-4]
        path = os.path.join(root,names[i])
        kp3d=get_3d_skeleton(path)
        np.save(os.path.join(save_keypoints_path,name)+".npy",kp3d)
