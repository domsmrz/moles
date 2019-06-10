import zipfile
import PIL.Image
import numpy as np
import pickle

USE_HSV = False
IMAGE_DIM = 224

with zipfile.ZipFile('skin-cancer-malignant-vs-benign.zip') as zip_handle:
    filenames = list(filter(lambda fn: fn.endswith('.jpg'), zip_handle.namelist()))
    data = np.zeros(shape=(len(filenames), IMAGE_DIM, IMAGE_DIM, 3), dtype=np.uint8)
    targets = np.zeros(shape=len(filenames), dtype=np.bool)
    for i_file, filename in enumerate(sorted(filenames)):
        with zip_handle.open(filename) as image_file_handle, PIL.Image.open(image_file_handle) as image_handle:
            if image_handle.size != (IMAGE_DIM, IMAGE_DIM):
                raise ValueError("Size of the image is as expected")

            if 'benign' in filename:
                target = False
            elif 'malignant' in filename:
                target = True
            else:
                raise ValueError("Invalid filename")
            targets[i_file] = target

            if USE_HSV:
                image_handle = image_handle.convert('HSV')
            pixels = image_handle.load()
            for x, y in np.ndindex((IMAGE_DIM, IMAGE_DIM)):
                data[i_file, x, y] = pixels[x, y]

with open('data{}.dat'.format('_HSV' if USE_HSV else ''), 'wb') as filehandle:
    pickle.dump(data, filehandle)
with open('targets{}.dat'.format('_HSV' if USE_HSV else ''), 'wb') as filehandle:
    pickle.dump(targets, filehandle)
