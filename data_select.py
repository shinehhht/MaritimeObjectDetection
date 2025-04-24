import os
from pathlib import Path
import shutil
import random

base_dir = Path(__file__).parent


def get_image_paths(directory, extensions={'jpg', 'jpeg', 'png', 'bmp'}): 
    return [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if os.path.splitext(f)[1][1:].lower() in extensions
    ]

def image_to_label_path(img_path):
    dir_path, filename = os.path.split(img_path)
    dir_parts = dir_path.split(os.sep)
    try:
        img_index = [p.lower() for p in dir_parts].index('images')
        dir_parts[img_index] = 'labels'
    except ValueError:
        raise ValueError("not found")

    new_dir = os.path.join(*dir_parts)
    
    name_without_ext = os.path.splitext(filename)[0]
    new_filename = f"{name_without_ext}.txt"
    
    return os.path.join(new_dir, new_filename)



#label_file_list = [image_to_label_path(x) for x in image_file_list]


# def extract_train_set(image_paths, label_paths, num,target_dir):
    
def copy_files(image_fog_list, train_dir):
    
    train_dst = Path(train_dir)
    # print(train_dir)
    train_dst.mkdir(parents=True, exist_ok=True)  # 创建目标目录
    
    train_dst_image = train_dst / f"images"
    # train_dst_label = train_dst / f"labels"
    train_dst_image.mkdir(parents=True, exist_ok=True)
    # train_dst_label.mkdir(parents=True, exist_ok=True)
    
    count_train = 0
    for src in map(Path, image_fog_list):  
        #print(f"src is {src}")
        if random.randint(0,2) < 2:
            print(src)
            if not src.exists():
                continue  # 跳过不存在的文件
            
        else:
            src = '/'.join(str(src).split('/')[:-3])+'/val_formal/images/'+src.name
            src = Path(src)
            
        train_dst_file = train_dst_image / src.name
        shutil.copy2(src, train_dst_file)
        count_train += 1
    print("done! count",count_train)
    
       
    
target_train_dir = str(base_dir)+"/train_with_mixed"
target_val_dir = str(base_dir)+"/val_with_mixed"
image_path_fog = str(base_dir)+'/val_with_fog/images'

image_fog_file_list = get_image_paths(image_path_fog)


copy_files(image_fog_file_list,target_val_dir)