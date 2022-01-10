import json

import numpy as np
import re
import tensorflow as tf


def get_hypers_model():
    with open("./config.json") as f:
        data = json.load(f)["model"]
    return data

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    return tensor
def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string
# Create necessary variables, lengths, and target
def make_variables(lines, properties,letters):
    sequence_and_length = [line2voc_arr(line,letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = tf.convert_to_tensor([sl[1] for sl in sequence_and_length], dtype = tf.int64)
    return pad_sequences(vectorized_seqs, seq_lengths, properties)

def line2voc_arr(line,letters):
    #print(line)
    arr = []
    regex = '(\[[^\[\]]{1,10}\])'
    line = replace_halogen(line)
    char_list = re.split(regex, line)
    #print("Char List: {}".format(char_list))
    for li, char in enumerate(char_list):
        if char.startswith('['):
               arr.append(letterToIndex(char,letters)) 
        else:
            chars = [unit for unit in char]

            for i, unit in enumerate(chars):
                arr.append(letterToIndex(unit,letters))
    
    return arr, len(arr)
def letterToIndex(letter,smiles_letters):
    return smiles_letters.index(letter)
# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths, properties):
    #print("Vec seq: {}".format(vectorized_seqs))
    #print("Init length: {}".format(seq_lengths))
    seq_tensor = np.zeros((len(vectorized_seqs), np.max(seq_lengths)), dtype = np.float32)
    #print("Shape of seq_tensor: {}".format(seq_tensor.shape))
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = seq #Get the array of index in vocabulary
    seq_tensor = tf.convert_to_tensor(seq_tensor, dtype=tf.float64)
    # Sort tensors by their length
    seq_lengths.numpy().sort()
    seq_lengths = seq_lengths[::-1]
    perm_idx = 0
    seq_tensor = seq_tensor[perm_idx]
    ##print("Seq length: {}".format(seq_tensor.shape[1]))
    # Also sort the target (countries) in the same order
    tmp_properties = int(properties)
    target = np.zeros((1,2))
    target[0][tmp_properties] = 1.0

    target = tf.convert_to_tensor(target, dtype=tf.float64)
    
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor),create_variable(seq_lengths),create_variable(target)
def pad_sequences_seq(vectorized_seqs, seq_lengths):
    seq_tensor = tf.zeros((len(vectorized_seqs), seq_lengths.max()), dtype = tf.float64)
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = tf.convert_to_tensor(seq,dtype=tf.float64)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
#     ##print(seq_tensor)
    seq_tensor = seq_tensor[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), create_variable(seq_lengths)

def construct_vocabulary(smiles_list,fname):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,10}\])'
        smiles = ds.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    ##print("Number of characters: {}".format(len(add_chars)))
    with open(fname, 'w') as f:
        f.write('<pad>' + "\n")
        for char in add_chars:
            f.write(char + "\n")
    return add_chars
def readLinesStrip(lines):
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    return lines
def getProteinSeq(path,contactMapName):
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    return seq
def getProtein(path,contactMapName,contactMap = True):
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    if(contactMap):
        contactMap = []
        for i in range(2,len(proteins)):
            contactMap.append(proteins[i])
        return seq,contactMap
    else:
        return seq

def getTrainDataSet(trainFoldPath):
    with open(trainFoldPath, 'r') as f:
        trainCpi_list = f.read().strip().split('\n')
    trainDataSet = [cpi.strip().split() for cpi in trainCpi_list]
    return trainDataSet#[[smiles, sequence, interaction],.....]
def getTestProteinList(testFoldPath):
    testProteinList = readLinesStrip(open(testFoldPath).readlines())[0].split()
    return testProteinList #['kpcb_2i0eA_full','fabp4_2nnqA_full',....]

def getSeqContactDict(contactPath,contactDictPath):# make a seq-contactMap dict 
    contactDict = open(contactDictPath).readlines()
    seqContactDict = {}
    for data in contactDict:
        _,contactMapName = data.strip().split(':')
        seq,contactMap = getProtein(contactPath,contactMapName)
        contactmap_np = [list(map(float, x.strip(' ').split(' '))) for x in contactMap]
        feature2D = np.expand_dims(contactmap_np, axis=0)
        feature2D = tf.convert_to_tensor(feature2D, dtype=tf.float64) #Shape = [1,H,W]
        seqContactDict[seq] = feature2D
    return seqContactDict
def getLetters(path):
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars
def getDataDict(testProteinList,activePath,decoyPath,contactPath):
    dataDict = {}
    for x in testProteinList:#'xiap_2jk7A_full'
        xData = []
        protein = x.split('_')[0]
        
        proteinActPath = activePath+"/"+protein+"_actives_final.ism"
        proteinDecPath = decoyPath+"/"+protein+"_decoys_final.ism"
        act = open(proteinActPath,'r').readlines()
        dec = open(proteinDecPath,'r').readlines()
        actives = [[x.split(' ')[0],1] for x in act] ######
        decoys = [[x.split(' ')[0],0] for x in dec]# test
        seq = getProtein(contactPath,x,contactMap = False)
        for i in range(len(actives)):
            xData.append([actives[i][0],seq,actives[i][1]])
        for i in range(len(decoys)):
            xData.append([decoys[i][0],seq,decoys[i][1]])
        ##print(len(xData))
        dataDict[x] = xData
    return dataDict

testFoldPath = './data/DUDE/dataPre/DUDE-foldTest3'
trainFoldPath = './data/DUDE/dataPre/DUDE-foldTrain3'
contactPath = './data/DUDE/contactMap'
contactDictPath = './data/DUDE/dataPre/DUDE-contactDict'
smileLettersPath  = './data/DUDE/voc/combinedVoc-wholeFour.voc'
seqLettersPath = './data/DUDE/voc/sequence.voc'
#print('get train datas....')
trainDataSet = getTrainDataSet(trainFoldPath)
#print('get seq-contact dict....')
seqContactDict = getSeqContactDict(contactPath,contactDictPath)
#print(seqContactDict)
#print('get letters....')
smiles_letters = getLetters(smileLettersPath)
sequence_letters = getLetters(seqLettersPath)
#count(trainFoldPath, contactDictPath)
#testProteinList = getTestProteinList(testFoldPath)# whole foldTest
#print("Test List: ", len(testProteinList))
# testProteinList = ['kpcb_2i0eA_full']# a protein of fold1Test

testProteinList = ['tryb1_2zebA_full','mcr_2oaxE_full', 'cxcr4_3oduA_full']# protein of fold3Test

DECOY_PATH = './data/DUDE/decoy_smile'
ACTIVE_PATH = './data/DUDE/active_smile'
#print('get protein-seq dict....')

#Get the contact map (pair-wise distance)
dataDict = getDataDict(testProteinList,ACTIVE_PATH,DECOY_PATH,contactPath)
#print("Length of testing",len(dataDict))

N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)
