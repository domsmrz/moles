import preprocessing
import sklearn.model_selection
import sklearn.metrics
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Model
from cnn import Network

DATA_DIR = 'data'

pipeline = [
    (preprocessing.extract_rgb, True),
    # (preprocessing.enrich_mirror, False),
]

data_file = os.path.join(DATA_DIR, 'data_train.dat')
targets_file = os.path.join(DATA_DIR, 'targets_train.dat')
data, targets = preprocessing.load_data(data_file), preprocessing.load_data(targets_file)

data_train, data_validation, targets_train, targets_validation = sklearn.model_selection.train_test_split(
    data, targets, test_size=0.25, random_state=42, shuffle=True, stratify=targets
)

for func, apply_test in pipeline:
    data_train, targets_train = func(data_train, targets_train)
    if apply_test:
        data_validation, targets_validation = func(data_validation, targets_validation)


print(data_train.shape[1:])
model = Network(input_shape=data_train.shape[1:])
model.fit(data_train, targets_train, data_validation, targets_validation)
predict = model.predict(data_validation)
print(sklearn.metrics.accuracy_score(targets_validation, predict))