import tensorflow as tf
from utils import * 
#[25921, [1,x], [1,H,W,1]]
"""

FoldTrain --> getTrainDataSet --> [...,[smiles, seq, label],...]
Get "contactDict" --> map from [seq] --> contactMap
x_train = [...,[smiles, contactMap],...]
y_train = [...,[label],...]

"""
def get_data_train(dataSet, contactDictionary):
    x_train, y_train = [], []
    for item in dataSet:
        lines, seq, proper = item
        contactMap = contactDictionary[seq]
        contactMap = np.reshape(contactMap, (1,contactMap.shape[1], contactMap.shape[-1],1))
        smiles, length, label = make_variables([lines], proper, smiles_letters)
        
        smiles = tf.reshape(smiles, [1, smiles.shape[-1]])
        x_train.append([contactMap, smiles])
        y_train.append(label)
        
    return x_train, y_train 


