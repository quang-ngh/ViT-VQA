import pickle
from utils import *
from dataTF import *
from tqdm import tqdm
from sklearn import metrics
import numpy as np

"""

"test_proteins": testProteinList
"testDataDict": dataDict

Create test data loader

"""

test_proteins = testProteinList
testDataDict = dataDict

def getROCE(predList,targetList,roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index,x] for index,x in enumerate(predList)]
    predList = sorted(predList,key = lambda x:x[1],reverse = True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce


def testModel(model, epoch):
    metric = {}
    for x in test_proteins:
        pred = []
        actual = []
        #Preparing data for Testing
        print("Current Testing -->", x.split('_')[0])
        data = dataDict[x]
        test_loader = get_data_test(data, seqContactDict)

        #Testing phase
        print("Starting Testing...")
        for lines, contactMap, proper in tqdm(test_loader):
            smiles, length, y = make_variables([lines], proper, smiles_letters)
            smiles = tf.reshape(smiles, [1, smiles.shape[-1]])
            
            logits = model(smiles, contactMap)
            #print("Predict: {} -- Actual: {}".format(np.argmax(logits), np.argmax(y)))
            #print("Predict: {} -- Actual: {}".format((logits), (y)))
            pred.append(np.argmax(logits))
            actual.append(np.argmax(y))
        
        f1_score = metrics.f1_score(actual, pred)
        recall = metrics.recall_score(actual, pred)
        precision = metrics.precision_score(actual, pred)
        acc = metrics.accuracy_score(actual, pred)
        print("F1: {} -- Recall :{} -- Precision: {} -- Accuracy: {}".format(f1_score, recall, precision, acc))
        metric['f1'] = f1_score
        metric['precision'] = precision
        metric['accuracy'] = acc
        metric['recall'] = recall

        print("Saving result...")
        inFile = open("metric_epoch"+str(epoch)+str(x)+".pkl", mode = 'wb')
        pickle.dump(metric, inFile)
        inFile.close()
        print("Saving Success!")

        print("End...")
        print("\n")

    

