from model import create_model
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    model = create_model()
    inputs = np.random.rand(1,420,420,1)
    outputs = model(inputs)
    print(outputs)
