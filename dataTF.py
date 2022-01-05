import tensorflow as tf
from utils import * 
import random
#[25921, [1,x], [1,H,W,1]]
"""

FoldTrain --> getTrainDataSet --> [...,[smiles, seq, label],...]
Get "contactDict" --> map from [seq] --> contactMap
x_train = [...,[smiles, contactMap],...]
y_train = [...,[label],...]

"""
class DrugDataSet(tf.keras.utils.Sequence):
    def __init__(self, dataset, seqcontactDict):
        random.shuffle(dataset)
        self.dataSet = dataset
        self.dict = seqcontactDict
        self.len = len(dataset)
        self.properties = [int(item[2]) for item in dataset]

    def __getitem__(self, index):
        smiles, seq, label = self.dataSet[index]
        contactMap = self.dict[seq]
        return smiles, contactMap, int(label)

    def __len__(self):
        return self.len

def get_data_train(dataSet, contactDictionary):
    train_loader = DrugDataSet(dataSet, contactDictionary)
    return train_loader

    


