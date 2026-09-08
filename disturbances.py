import numpy as np


def random_torque(model, magnitude=3.0):
    """Random disturbance torque, one value per joint (size model.nv),
    uniformly sampled in [-magnitude, magnitude].
    """
    return np.random.uniform(-magnitude, magnitude, size=model.nv)
