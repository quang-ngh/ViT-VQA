from math import e
from random import random, uniform
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow.python.ops.array_ops import zeros
from core.embedding import *
from core.encode import *
from utils import *
import numpy as np

class MCB(tf.keras.layers.Layer):
    def __init__(self, h_vec, s_vec, output_dim):
        super(MCB, self).__init__()
        self.h_vec = tf.cast(tf.reshape(h_vec, (h_vec.shape[-1],)), dtype=tf.float32)
        self.s_vec = tf.cast(tf.reshape(s_vec, (s_vec.shape[-1],)), dtype=tf.float32)
        self.output_dim = output_dim

    def call(self, img_vec, seq_vec):
        output_dim = self.output_dim
        img_shape = img_vec.shape[-1]
        seq_shape = seq_vec.shape[-1]
        
        #for i in range(output_dim)
        #projection[h[i]] = s_vec[i]*vec_passed[i]
        indices_img = np.concatenate((np.arange(img_shape)[..., np.newaxis],
                              self.h_vec[..., np.newaxis]), axis=1)
        
        indices_seq = np.concatenate((np.arange(seq_shape)[..., np.newaxis],
                              self.h_vec[..., np.newaxis]), axis=1)
        
        print(indices_img)

        sketch_seq = tf.sparse.reorder(
            tf.SparseTensor(indices_seq, self.s_vec,[seq_vec.shape[-1], output_dim])
        )
                
        sketch_img = tf.sparse.reorder(
            tf.SparseTensor(indices_img, self.s_vec,[seq_vec.shape[-1],output_dim])
        )

        img_flat = tf.reshape(img_vec, [-1, img_shape])
        seq_flat = tf.reshape(seq_vec, [-1, seq_shape])

        #Vector represent for count sketch projection of feature 2D and string inputs
        #After projection
        img_vec = tf.transpose(tf.sparse.sparse_dense_matmul(sketch_img, img_flat, adjoint_a = True, adjoint_b = True))
        seq_vec = tf.transpose(tf.sparse.sparse_dense_matmul(sketch_seq, seq_flat, adjoint_a = True, adjoint_b = True))

        fft_img = tf.signal.fft(tf.complex(real = img_vec, imag = tf.zeros_like(img_vec)))
        fft_seq = tf.signal.fft(tf.complex(real = seq_vec, imag = tf.zeros_like(seq_vec)))

        output = tf.multiply(fft_img, fft_seq) #Element-wise product
    
        output = tf.math.real(tf.signal.ifft(output))
        output = tf.math.l2_normalize(output, epsilon = 1e-5)
        
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
        
        self.h_vector = tf.random.uniform(shape = (1,Dim), minval=0, maxval=mcb_dim)
        self.s_vector = tf.random.uniform(shape = (1,Dim), minval=0, maxval=2)
        self.h_vector = tf.cast(self.h_vector, tf.int32)
        #print("h vec: {}".format(self.h_vector))
        self.s_vector = tf.cast(tf.floor(self.s_vector)*2-1, tf.int32)
        #asprint("S vec: {}".format(self.s_vector))

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
        
        img_rep = self.encoderV(v_embd) #Shape = [batch_size, Dim] 
        seq_rep = self.encoderL(l_embd)

        img_vec = img_rep[:, 0]
        seq_vec = seq_rep[:, 0]
        mcb_output = self.mcb(img_vec, seq_vec, 2048)
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
