import numpy as np
import os
import cv2
import math
from numba import jit
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
base_dir = Path(__file__).parent.parent

train_images_path = str(base_dir)+'/data/train_coco_hybrid/images'
val_images_path = str(base_dir)+'/data/val_coco_hybrid/images'
test_images_path = str(base_dir)+'/data/test/images'

def get_image_paths(directory, extensions={'jpg', 'jpeg', 'png', 'bmp'}): 
    return [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if os.path.splitext(f)[1][1:].lower() in extensions
    ]


def process(image_path):
    img_name = image_path.split('/')[-1]
    if os.path.exists(str(base_dir)+'/data/val_with_fog_coco/images/' + img_name):
        print("have built")
        return 
    print(img_name)
    
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = cv2.imread(image_path)
    i = random.randint(0,10)
    
    @jit()
    def AddHaz_loop(img_f, center, size, beta, A):
        (row, col, chs) = img_f.shape

        for j in range(row):
            for l in range(col):
                d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-beta * d)
                img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
        return img_f

    img_f = image/255
    (row, col, chs) = image.shape
    A = 0.5  
    # beta = 0.08  
    beta = 0.01 * i + 0.05
    size = math.sqrt(max(row, col)) 
    center = (row // 2, col // 2)  
    foggy_image = AddHaz_loop(img_f, center, size, beta, A)
    img_f = np.clip(foggy_image*255, 0, 255)
    img_f = img_f.astype(np.uint8)
    #img_name = str(image_dir)+'/vdd/data_vocfog/train/JPEGImages/' + image_name \
    #          + '_' + ("%.2f"%beta) + '.' + image_name_index
    img_name = str(base_dir)+'/data/val_with_fog_coco/images/' + img_name
    print("path is ",img_name)
    cv2.imwrite(img_name, img_f)
    print("done once")


"""
for image_path in tqdm(image_paths):
    process(image_path)
"""
def main():
    image_paths = get_image_paths(val_images_path)
    num_threads = os.cpu_count() * 2  # 设置线程数（建议为CPU核心数的2倍）

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 使用tqdm显示进度条，并并行处理任务
        list(tqdm(executor.map(process, image_paths), total=len(image_paths)))

if __name__ == "__main__":
    main()