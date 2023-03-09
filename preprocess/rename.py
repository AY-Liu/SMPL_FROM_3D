import os  

path="/mnt/disk_2/aoyang/new_txt"  
for root,_,file in os.walk(path):
    break
file = sorted(file)
for i in range(len(file)):
    name = file[i][8:]
    # name = "%06d"%(int(file[i][:-4])-1)+".jpg"
    # name = str(file[i])+".jpg"
    # print(name)
    os.rename(os.path.join(root,file[i]),os.path.join(root,name))
