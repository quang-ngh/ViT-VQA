import tensorflow as tf

class FC(tf.keras.layers.Layer):
    def __init__(self, dense_layers, dropout, activation = 'gelu'):
        super(FC,self).__init__()
        
        layers = []
        for num_of_units in dense_layers:
            layers.extend(
                [tf.keras.layers.Dense(units = num_of_units, activation = activation),
                tf.keras.layers.Dropout(0.1)]
            )
        
        self.FC = tf.keras.Sequential(layers)
    
    def call(self, inputs, *args, **kwargs):
        output = self.FC(inputs, *args, **kwargs)
        return output

class MHSABlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, Dim, hidden_layers, dropout, norm_coff = 1e-12):
        super(MHSABlock, self).__init__()
        
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = Dim, dropout = dropout
        )
        self.FC = FC(hidden_layers, dropout)
        self.layerNormAtt = tf.keras.layers.LayerNormalization(epsilon = norm_coff)
        self.layerNormFC = tf.keras.layers.LayerNormalization(epsilon = norm_coff)
    
    def call(self, inputs):
        norm_attention = self.layerNormAtt(inputs) #Pass by layer norm
        
        attention = self.attention(query = norm_attention, value = norm_attention) # Pass by Multihead attention
        attention += inputs #Skip connection
        
        output = self.layerNormFC(attention) # Pass by layer norm 
        output = self.FC(output) # Pass by fully connected 
        
        output += attention # Skip connection
        return output

class VEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers_encoder, num_heads, Dim, hidden_dim, dropout, norm_coff = 1e-12):
        super(VEncoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                MHSABlock(
                    num_heads = num_heads,
                    Dim = Dim, 
                    hidden_layers=[hidden_dim, Dim],
                    dropout = dropout,
                    norm_coff=norm_coff
                )
                for _ in range(num_layers_encoder)
            ]
        )
    def call(self, inputs, *args, **kwargs):
        output = self.encoder(inputs)
        return output
