from random import random, uniform
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow.python.ops.array_ops import zeros
from core.embedding import *
from core.encode import *
from utils import *
import numpy as np

class MCB(tf.keras.layers.Layer):
    def __init__(self, h_vec, s_vec):
        super(MCB, self).__init__()
        self.h_vec = h_vec
        self.s_vec = s_vec
    
    def call(self, img_vec, seq_vec):
        #for i in range(img_vec.shape[-1]):
        #    index = self.h_vec[0][i]
        #    img_vec[0][index] = self.s_vec[0][i] * img_vec[0][i]
        #for i in range(seq_vec.shape[-1]):
        #    index = self.h[0][i]
        #    seq_vec[0][index] = self.s_vec[0][i] * seq_vec[0][i]
        
        img_vec_fft = tf.cast(img_vec, dtype = tf.complex64)
        seq_vec_fft = tf.cast(seq_vec, dtype = tf.complex64)
        img_vec_fft = tf.signal.fft2d(img_vec_fft)
        seq_vec_fft = tf.signal.fft2d(seq_vec_fft)

        output = img_vec_fft * seq_vec_fft
        output = tf.signal.irfft2d(output)
        output = tf.math.l2_normalize(output, epsilon = 1e-3)
        
        return output

class MHSADrugVQA(tf.keras.models.Model):
    def __init__(self, num_layers, num_heads, Dim, hidden_dim, dropout, patch_size, n_chars, norm_coff, mcb_dim = 800):
        super(MHSADrugVQA, self).__init__()
        self.encoderV = Encoder(num_layers_encoder=num_layers,
                                num_heads = num_heads,
                                Dim = Dim,
                                hidden_dim = hidden_dim,
                                dropout = dropout,
                                norm_coff = norm_coff)
        self.encoderL = Encoder(num_layers_encoder=num_layers,
                                num_heads = num_heads,
                                Dim = Dim,
                                hidden_dim = hidden_dim,
                                dropout = dropout,
                                norm_coff = norm_coff)

        self.Lembedding = Smiles_Embedding(n_chars, Dim)
        self.Vembedding = PatchesEmbedding(patch_size, Dim)
        
        self.h_vector = np.random.random_integers(low = 0, high = mcb_dim, size = (1,Dim))
        self.s_vector = np.random.choice([1.0,-1.0], size = (1,Dim))

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon = norm_coff),
            tf.keras.layers.Dense(units = hidden_dim, activation = 'tanh'),
            tf.keras.layers.Dropout(rate = dropout),
            tf.keras.layers.Dense(units = 2, activation = 'softmax')
        ]
        )
        self.mcb = MCB(self.h_vector, self.s_vector)
        #self.flatten = tf.keras.layers.Flatten()
        #self.dense = tf.keras.layers.Dense(units = 2, activation = 'softmax')
        #self.poolV = tf.keras.layers.MaxPool2D()

    def call(self, smiles, contactMap):
        
        #Processing 2D Feature
        smiles = tf.Variable(smiles, dtype=tf.float32)
        contactMap = tf.Variable(contactMap, dtype = tf.float32)
        smiles = tf.reshape(smiles, (1,tf.shape(smiles)[-1],1))

        v_embd = self.Vembedding(contactMap)
        l_embd = self.Lembedding(smiles)
        #print("Shape of inp V: {}".format(tf.shape(v_embd)))
        #print("Shape of inp L: {}".format(tf.shape(l_embd)))
        img_rep = self.encoderV(v_embd) #Shape = [batch_size, Dim] 
        seq_rep = self.encoderL(l_embd)
        
        #img_vec = img_rep[:,0]#self.flatten(img_rep)
        #seq_vec = seq_rep[:, 0]#self.flatten(seq_rep)
        #img_vec = tf.reshape(tf.nn.max_pool1d(img_rep, ksize = (img_rep.shape[1]), strides = 1, padding = "VALID"), (1,-1)) #add batch_size 
        #seq_vec = tf.reshape(tf.nn.max_pool1d(seq_rep, ksize=(seq_rep.shape[1]), strides = 1, padding = "VALID"), (1,-1)) #add batch_size
        

        img_vec = img_rep[:, 0]
        seq_vec = seq_rep[:, 0]
        print("Shape: {}".format(tf.shape(img_vec)))
        print("Shape: {}".format(tf.shape(seq_vec)))
        mcb_output = self.mcb(img_vec, seq_vec)
        #print("Shape of mcb output: {}".format(tf.shape(mcb_output)))
        #mcb_output = tf.concat([img_vec, seq_vec], axis = 1)
        #print("Inp shape : {}".format(tf.shape(mcb_output)))
        output = self.classifier(mcb_output)
        

        return output

def create_model():
    args = get_hypers_model()
    model = MHSADrugVQA(
        num_layers=args["num_layers"],
        num_heads=args["num_head"],
        Dim = args["dimension"],
        hidden_dim=args["dense_units"],
        dropout=args["dropout"],
        patch_size=args["patch_size"],
        norm_coff=args["norm_coff"],
        n_chars=args["n_chars"]
    )
    return model
