import zipfile
import PIL.Image
import numpy as np
import pickle
import sklearn.model_selection

IMAGE_DIM = 224

with zipfile.ZipFile('skin-cancer-malignant-vs-benign.zip') as zip_handle:
    filenames = list(filter(lambda fn: fn.endswith('.jpg'), zip_handle.namelist()))
    data = np.zeros(shape=(len(filenames), IMAGE_DIM, IMAGE_DIM, 6), dtype=np.uint8)
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

            pixels_rgb = image_handle.load()
            image_handle_hsv = image_handle.convert('HSV')
            pixels_hsv = image_handle_hsv.load()
            for x, y in np.ndindex((IMAGE_DIM, IMAGE_DIM)):
                data[i_file, x, y] = pixels_rgb[x, y] + pixels_hsv[x, y]

data_train, data_test, targets_train, targets_test = sklearn.model_selection.train_test_split(
        data, targets, test_size=0.2, random_state=42, shuffle=True, stratify=targets)

with open('data_train.dat', 'wb') as filehandle:
    pickle.dump(data_train, filehandle)
with open('targets_train.dat', 'wb') as filehandle:
    pickle.dump(targets_train, filehandle)
with open('data_test.dat', 'wb') as filehandle:
    pickle.dump(data_test, filehandle)
with open('targets_test.dat', 'wb') as filehandle:
    pickle.dump(targets_test, filehandle)
