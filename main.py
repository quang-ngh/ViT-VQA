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

def train(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_obj = tf.keras.losses.BinaryCrossentropy()
    dataset = get_data_train(trainDataSet, seqContactDict)

    for epoch in range(EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()
        print(epoch)
        for batch, (lines, contactMap, proper) in enumerate(dataset):
            contactMap = np.reshape(contactMap, (1,contactMap.shape[1], contactMap.shape[-1],1))
            smiles, length, y = make_variables([lines], proper, smiles_letters)
            smiles = tf.reshape(smiles, [1, smiles.shape[-1]])

            y_hat = model(smiles, contactMap)
            loss =loss_obj(y, y_hat)
            with tf.GradientTape() as tape:
                grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss)
        train_loss.append(epoch_loss_avg.result())
        if epoch % 10:
            print("Loss: {}".format(epoch_loss_avg.result()))

train(model)   
#datasets = get_data_train(trainDataSet, seqContactDict)
#for batch, (lines, mapp, proper) in enumerate(datasets):
#    print("Lines :{} - Map :{}".format(lines, mapp))
                