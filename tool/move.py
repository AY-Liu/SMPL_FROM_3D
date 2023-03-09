import os
import shutil
from tqdm import tqdm

def copy_files(A, B):
    for root, dirs, files in os.walk(A):
        for file in tqdm(files, desc="Copying files"):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(B, file)
            if os.path.exists(dst_file):
                print(f"Overwriting file: {dst_file}")
            shutil.copy2(src_file, dst_file)
            
if __name__ == '__main__':
    A = "/mnt/disk_2/aoyang/new_txt"
    B = "/mnt/disk_2/aoyang/FLAG3D/skeleton"
    copy_files(A,B)