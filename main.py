from core.model import create_model
import numpy as np
import tensorflow as tf
import random


model = create_model()
vinputs = np.random.rand(1,420,420,1)
linputs = np.random.rand(1, random.randint(50,100))
outputs = model(vinputs, linputs)
