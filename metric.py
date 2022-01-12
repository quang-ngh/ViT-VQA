import pickle
import numpy as np
from core.model import create_model 
import tensorflow as tf

from utils import create_variable
def get_metric(path):
    outFile = open(path, mode = 'rb')
    metric = pickle.load(outFile)
    outFile.close()
    print(metric)

mymodel = create_model()
mymodel.load_weights("/home/nhqcs/Desktop/Github/DrugDesign/mymodel.index")
print("load sucess")
path = "/home/nhqcs/Desktop/Github/DrugDesign/metric_epoch0tryb1_2zebA_full.pkl"
get_metric(path)