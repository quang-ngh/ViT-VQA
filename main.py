from tensorflow.keras import datasets
from core.model import MHSADrugVQA, create_model
import numpy as np
import tensorflow as tf
import random
from dataTF import *
from utils import *
EPOCHS = 100
train_loss = []
model = create_model()
def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_obj = tf.keras.losses.BinaryCrossentropy()
    x_train, y_train = get_data_train(trainDataSet, seqContactDict)

    for epoch in range(EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()

        for i in range(len(x_train)):
            x = x_train[i]
            y = y_train[i]
            print(y)
            y_hat = model(x, training = True) #Predict
            loss = loss_obj(y, y_hat) #Calculate loss
            with tf.GradientTape() as tape:
                grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))

            epoch_loss_avg.update_state(loss)
        train_loss.append(epoch_loss_avg.result())
        if epoch % 50 == 0:
            print("Loss: {}".format(epoch_loss_avg.result()))

train(model)
                