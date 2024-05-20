import numpy as np
import imageio
from PIL import Image
import os
########################################################################
def read_image(file_path):
    img_in = np.asarray(Image.open(file_path))
    img_in = np.transpose(img_in, (2,0,1))

    return img_in

########################################################################
def write_image(file_path, array):
    array = np.transpose(array, (1, 2, 0))
    imageio.imwrite(file_path , array.astype(np.uint8))

########################################################################
def RGB_generate(folder_path, img_list):
    image_files = [f for f in os.listdir(folder_path) if not f.startswith("RGGB") and f.endswith(".png")]
    image_files.sort(key = lambda x : int(x.split(".")[0]))
    result = np.zeros_like(read_image(os.path.join(folder_path, image_files[0])))
    count = 0
    for i in img_list:
        img = read_image(os.path.join(folder_path, image_files[i]))
        count += 1
        result += img
    print('Done! ', folder_path)
    result = result /count
    result = result / np.max(result) *255
    output_file_path = os.path.join(folder_path, "RGGB.png")
    write_image(output_file_path, result.astype(np.float64))

if __name__ == '__main__':
    path = '../data/RGB/result'
    folders = os.listdir(path)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        img_list = [0,22,57]
        wavelength = np.linspace(470,900,150)
        #print(np.ceil(wavelength[img_list]))
        RGB_generate(folder_path, img_list)