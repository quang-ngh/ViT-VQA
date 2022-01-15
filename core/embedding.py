import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
class Patches(tf.keras.layers.Layer):
    def __init__(self, pat_size):
        super(Patches, self).__init__()
        self.patches_size = pat_size
    
    def call(self, contactMap):
        batch_size = tf.shape(contactMap)[0]
    
        patches = tf.image.extract_patches(
            images = contactMap,
            sizes = [1, self.patches_size, self.patches_size, 1],
            rates = [1,1,1,1],
            strides = [1, self.patches_size, self.patches_size,1],
            padding = "VALID"
        )
        
        dimension = patches.shape[-1]
        
        patches = tf.reshape(patches, (batch_size, -1, dimension))
        
        return patches

class PatchesEmbedding(tf.keras.layers.Layer):
    """
    Input: 
        Feature 2D: a tensor 4D [batch_size, height, width, channels]
        Output: Patches embedding --> a tensor 4D [batch_size, num_patches, num_patches, H*W]
    """
    def __init__(self, patch_size, hidden_dim):
        super(PatchesEmbedding, self).__init__()
        
        self.patches_size = patch_size
        self.patches = Patches(patch_size)
        self.hidden_dim = hidden_dim
        self.projection = tf.keras.layers.Dense(units = hidden_dim)
        
        self.cls_token = self.add_weight(
            "cls_token",
            shape = [1,1,hidden_dim],
            initializer = tf.keras.initializers.RandomNormal(),
            dtype = tf.float32
        )
    def call(self, contactMap):
        
        #Padding ContactMap
        #print("Shape of 2D: {}".format(tf.shape(contactMap)))
        #print("Shape before padding: {}".format(tf.shape(contactMap)))
        contactMap_size = tf.shape(contactMap)[1]
        #print("Shape of contactMap: {}".format(contactMap_size))
        self.num_patches = (contactMap_size // self.patches_size) ** 2
        #print("Num of patches: {}".format(self.num_patches))
        self.pos_embedding = self.add_weight(
            "pos_embd",
            shape = [self.num_patches + 1, self.hidden_dim],
            initializer = tf.keras.initializers.RandomNormal(),
            dtype = tf.float32
        )
        
        patches = self.patches(contactMap)
        #fig = plt.figure(figsize = (self.patches_size,self.patches_size))
        
        patches_encoded = self.projection(patches)
        
        tmp_cls = tf.cast(
            tf.broadcast_to(self.cls_token, [tf.shape(contactMap)[0],1,tf.shape(patches_encoded)[-1]]),
            dtype = contactMap.dtype
        )
        
        patches_encoded = tf.concat([tmp_cls, patches_encoded], axis = 1)
        patches_encoded = patches_encoded + self.pos_embedding
        
        #print("Shape {}".format(patches_encoded.shape))
        return patches_encoded

class Smiles_Embedding(tf.keras.layers.Layer):
    def __init__(self, n_char, hidden_dim):
        super(Smiles_Embedding, self).__init__()
        self.n_char = n_char
        self.hidden_dim = hidden_dim
        self.embd = tf.keras.layers.Embedding(n_char, hidden_dim, name = "smiles_embd")
        self.s_cls_token = self.add_weight(
            name = "string_cls_token",
            shape = [1,1,hidden_dim],
            initializer = tf.keras.initializers.RandomNormal(),
            dtype = tf.float32
        )

    def call(self, inputs):
        #print("Shape of string inp: {}".format(tf.shape(inputs)))
        output = self.embd(inputs)
        tmp_cls = tf.cast(tf.broadcast_to(self.s_cls_token, [tf.shape(inputs)[0],1,tf.shape(output)[-1]]),
                            dtype = tf.float32)
        output = tf.concat([tmp_cls, output], axis = 1)
        return output
