import tensorflow as tf
from core.embedding import *
from core.encode import *
from utils import *

class MHSADrugVQA(tf.keras.models.Model):
    def __init__(self, num_layers, num_heads, Dim, hidden_dim, dropout, patch_size, n_chars=217, norm_coff = 1e-12):
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
        self.dense = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon = norm_coff),
            tf.keras.layers.Dense(units = hidden_dim),
            tf.keras.layers.Dropout(rate = dropout),
            tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
        ]
        )
        self.Lembedding = tf.keras.layers.Embedding(n_chars, Dim)
        self.flatten = tf.keras.layers.Flatten()


    def call(self, contactMap, smiles):
        #Processing 2D Feature
        v_embd = self.Vembedding(contactMap)
        l_embd = self.Lembedding(smiles)
        print("Shape of smiles: {}".format(tf.shape(l_embd)))
        img_rep = self.encoderV(v_embd) #Shape = [batch_size, Dim]
        seq_rep = self.encoderL(l_embd)
        img_vec = self.flatten(img_rep[:,0])
        
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