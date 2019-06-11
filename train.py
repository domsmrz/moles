import preprocessing
import sklearn.model_selection
import sklearn.metrics
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Model

DATA_DIR = '.'

pipeline = [
    (preprocessing.extract_rgb, True),
    (preprocessing.enrich_mirror, False),
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

# TO BE CHANGED
data_train = np.reshape(data_train, newshape=(data_train.shape[0], -1))
data_validation = np.reshape(data_validation, newshape=(data_validation.shape[0], -1))

model = Model(max_depth=2)
model.fit(data_train, targets_train)

predictions = model.predict(data_validation)
acc = sklearn.metrics.accuracy_score(targets_validation, predictions)
print("Acc: {}".format(acc))