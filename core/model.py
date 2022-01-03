import tensorflow as tf
from embedding import *
from encode import *
from utils import *

class MHSADrugVQA(tf.keras.models.Model):
    def __init__(self, num_layers, num_heads, Dim, hidden_dim, dropout, patch_size, norm_coff = 1e-12):
        super(MHSADrugVQA, self).__init__()
        self.embedding = PatchesEmbedding(patch_size, Dim)
        self.encoderV = VEncoder(num_layers_encoder=num_layers,
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


    def call(self, contactMap):
        #Processing 2D Feature
        embd = self.embedding(contactMap)
        img_rep = self.encoderV(embd) #Shape = [batch_size, Dim]

        #Processing SMILES

        return img_rep

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