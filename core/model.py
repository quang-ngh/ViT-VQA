from random import random, uniform
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow.python.ops.array_ops import zeros
from core.embedding import *
from core.encode import *
from utils import *
import numpy as np


class MHSADrugVQA(tf.keras.models.Model):
    def __init__(self, num_layers, num_heads, Dim, hidden_dim, dropout, patch_size, mcb_dim = 800, n_chars=217, norm_coff = 1e-12):
        super(MHSADrugVQA, self).__init__()
        self.Vembedding = PatchesEmbedding(patch_size, Dim)
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
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon = norm_coff),
            tf.keras.layers.Dense(units = hidden_dim),
            tf.keras.layers.Dropout(rate = dropout),
            tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
        ]
        )
        self.Lembedding = tf.keras.layers.Embedding(n_chars, Dim)
        self.flatten = tf.keras.layers.Flatten()
        self.h_vector = tf.convert_to_tensor(np.random.random_integers(low = 0, high = mcb_dim, size = (1,Dim)))
        self.s_vector = tf.convert_to_tensor(np.random.choice([1,-1], size = (1,Dim)))
        print(self.h_vector)
        print(self.s_vector)

    def CS_projection(self,v,dim):
        projection = tf.zeros((1,dim), dtype= tf.float32)
        for i in range(dim):
            projection[self.h_vector[i]] = projection[self.h_vector[i]] + self.s_vector[i]*v[i]
        return projection
    
    def MCB(self,img_vec, seq_vec):
        img_proj = self.CS_projection(img_vec, dim=tf.shape(img_vec)[-1])
        seq_proj = self.CS_projection(seq_vec, tf.shape(seq_vec)[-1])

        output = tf.signal.fft(img_proj) * tf.signal.fft(seq_proj)
        output = tf.signal.ifft(output)


    def call(self, contactMap, smiles):
        #Processing 2D Feature
        v_embd = self.Vembedding(contactMap)
        l_embd = self.Lembedding(smiles)
        
        img_rep = self.encoderV(v_embd) #Shape = [batch_size, Dim]
        seq_rep = self.encoderL(l_embd)
        
        img_vec = img_rep[:,0]
        seq_vec = seq_rep[:,0]

        mcb_output = self.MCB(img_vec, seq_vec)
        print("MCB Shape: {}".format(tf.shape(mcb_output)))



        #Processing SMILES

        return img_rep, seq_rep

def create_model():
    args = get_hypers_model()
    model = MHSADrugVQA(
        num_layers=args["num_layers"],
        num_heads=args["num_head"],
        Dim = args["dimension"],
        hidden_dim=args["dense_units"],
        dropout=args["dropout"],
        patch_size=args["patch_size"],
        norm_coff=args["norm_coff"]
    )
    return model