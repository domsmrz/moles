import pickle
import numpy as np


def load_data(data):
    if isinstance(data, str):
        with open(data, 'rb') as f:
            data = pickle.load(f)
    elif hasattr(data, 'read'):
        data = pickle.load(data)
    return data


def extract_rgb(data, targets=None):
    data = load_data(data)
    new_data = data[:, :, :, :3]
    if targets is None:
        return new_data
    return new_data, targets


def extract_hsv(data, targets=None):
    data = load_data(data)
    new_data = data[:, :, :, 3:]
    if targets is None:
        return new_data
    return new_data, targets


def enrich_hsv_rotation(data, targets=None, rotations=4):
    data = load_data(data)
    shape = list(data.shape)
    shape[-1] += rotations - 1
    new_data = np.zeros(shape=shape, dtype=np.int8)
    new_data[:, :, :, :3] = data[:, :, :, :3]
    new_data[:, :, :, -2:] = data[:, :, :, 4:]

    rotation_angle = 0x100 // rotations
    for i in range(rotations):
        new_data[:, :, :, 3+i] = data[:, :, :, 3] + ((i * rotation_angle) & 0xff)

    if targets is None:
        return new_data
    return new_data, targets


def enrich_mirror(data, targets, shuffle=True, seed=42):
    data = load_data(data)
    targets = load_data(targets)
    assert data.shape[0] == targets.shape[0]

    mirror_1 = np.flip(data, axis=1)
    mirror_2 = np.flip(data, axis=2)
    mirror_12 = np.flip(mirror_1, axis=2)
    new_data = np.concatenate((data, mirror_1, mirror_2, mirror_12), axis=0)
    new_targets = np.concatenate((targets, targets, targets, targets))

    if not shuffle:
        return new_data, new_targets

    permutation = np.random.RandomState(seed=seed).permutation(new_data.shape[0])
    return new_data[permutation, :, :, :], new_targets[permutation]
