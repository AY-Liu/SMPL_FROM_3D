import os  

path="/mnt/disk_2/aoyang/new_imgs"  
for root,file,_ in os.walk(path):
    break
for i in range(len(file)):
    name = file[i][8:-4]
    # print(name)
    os.rename(os.path.join(root,file[i]),os.path.join(root,name))  
