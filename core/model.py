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
        self.h_vector = np.random.random_integers(low = 0, high = mcb_dim, size = (1,Dim))
        self.s_vector = np.random.choice([1.0,-1.0], size = (1,Dim))
        
    def CS_projection(self,v,dim):
        projection = np.zeros((1,dim), dtype=np.float32)
        
        for i in range(dim):
            
            projection[0][self.h_vector[0][i]] = projection[0][self.h_vector[0][i]] + self.s_vector[0][i]*v[0][i]
        return tf.convert_to_tensor(projection, dtype=tf.float32)
    
    def MCB(self,img_vec, seq_vec):
        img_proj = tf.cast(self.CS_projection(img_vec, dim=img_vec.shape[-1]), dtype=tf.complex64)
        seq_proj = tf.cast(self.CS_projection(seq_vec, seq_vec.shape[-1]), dtype=tf.complex64)

        output = tf.signal.fft2d(img_proj) * tf.signal.fft2d(seq_proj)
        output = tf.signal.irfft2d(output)
        return output

    def call(self, contactMap, smiles):
        #Processing 2D Feature
        v_embd = self.Vembedding(contactMap)
        l_embd = self.Lembedding(smiles)
        
        img_rep = self.encoderV(v_embd) #Shape = [batch_size, Dim]
        seq_rep = self.encoderL(l_embd)
        
        img_vec = img_rep[:,0]
        seq_vec = seq_rep[:,0]
        
        mcb_output = self.MCB(img_vec, seq_vec)
        
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
        norm_coff=args["norm_coff"]
    )
    return model