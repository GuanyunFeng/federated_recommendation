from time import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras import initializers,regularizers,optimizers
from tensorflow.python.ops import clip_ops
from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.backend as K
from Dataset import Dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_compiled_gmf_model(num_users, num_items, latent_dim=8, learning_rate = 0.001, reg = [0, 0]):
    # Input variables
    user_input = layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = layers.Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding = layers.Embedding(input_dim=num_users, output_dim=latent_dim,
                                  embeddings_initializer=initializers.RandomNormal(),
                                  embeddings_regularizer=regularizers.l2(reg[0]),
                                  input_length=1, name='user_embedding')
    item_embedding = layers.Embedding(input_dim=num_items, output_dim=latent_dim,
                                  embeddings_initializer=initializers.RandomNormal(),
                                  embeddings_regularizer=regularizers.l2(reg[1]),
                                  input_length=1, name='item_embedding')
    
    # Crucial to flatten an embedding vector!
    user_latent = layers.Flatten()(user_embedding(user_input))
    item_latent = layers.Flatten()(item_embedding(item_input))
    
    # Element-wise product of user and item embeddings
    predict_vector = layers.multiply([user_latent, item_latent])
    prediction = layers.Dense(1, activation='sigmoid',
                kernel_initializer=initializers.lecun_normal(), name='prediction')(predict_vector)

    model_gmf = models.Model(inputs=[user_input, item_input], outputs=prediction)
    model_gmf.compile(optimizer=optimizers.Adam(lr=learning_rate, clipnorm=0.5), loss='binary_crossentropy')
    #model_gmf.compile(optimizer=Accoptimizers.Adam(lr=learning_rate, clipnorm=0.5), loss='binary_crossentropy')

    return model_gmf




def get_compiled_mlp_model(num_users, num_items, learning_rate = 0.001, layers_num=[20, 10], reg_layers=[0, 0]):
    assert len(layers_num) == len(reg_layers)
    num_layer = len(layers_num)  # Number of layers in the MLP
    # Input variables
    user_input = layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = layers.Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding = layers.Embedding(input_dim=num_users, output_dim=int(layers_num[0]/2), name='user_embedding',
                                   embeddings_regularizer=regularizers.l2(reg_layers[0]), input_length=1)
    item_embedding = layers.Embedding(input_dim=num_items, output_dim=int(layers_num[0]/2), name='item_embedding',
                                   embeddings_regularizer=regularizers.l2(reg_layers[0]), input_length=1)
    
    # Crucial to flatten an embedding vector!
    user_latent = layers.Flatten()(user_embedding(user_input))
    item_latent = layers.Flatten()(item_embedding(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    vector = layers.concatenate([user_latent, item_latent])
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = layers.Dense(layers_num[idx], kernel_regularizer=regularizers.l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = layers.Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_normal(),
                       name='prediction')(vector)
    
    model_mlp = models.Model(inputs=[user_input, item_input], outputs=prediction)
    model_mlp.compile(optimizer=optimizers.Adam(lr=learning_rate, clipnorm=0.5), loss='binary_crossentropy')
    
    return model_mlp




def get_compiled_neumf_model(num_users, num_items, learning_rate = 0.001, mf_dim=10, layers_num=[10], reg_layers=[0], reg_mf=0):
    assert len(layers_num) == len(reg_layers)
    num_layer = len(layers_num) #Number of layers in the MLP
    # Input variables
    user_input = layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = layers.Input(shape=(1,), dtype='int32', name='item_input')
    
    # Embedding layer
    mf_embedding_user = layers.Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_initializer=initializers.RandomNormal(),
                                  embeddings_regularizer=regularizers.l2(reg_mf),
                                  input_length=1)
    mf_embedding_item = layers.Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_initializer=initializers.RandomNormal(),
                                  embeddings_regularizer=regularizers.l2(reg_mf), input_length=1)

    mlp_embedding_user = layers.Embedding(input_dim=num_users, output_dim=int(layers_num[0]/2), name="mlp_embedding_user",
                                   embeddings_initializer=initializers.RandomNormal(),
                                   embeddings_regularizer=regularizers.l2(reg_layers[0]), input_length=1)
    mlp_embedding_item = layers.Embedding(input_dim=num_items, output_dim=int(layers_num[0]/2), name='mlp_embedding_item',
                                   embeddings_initializer=initializers.RandomNormal(),
                                   embeddings_regularizer=regularizers.l2(reg_layers[0]), input_length=1)
    
    # MF part
    mf_user_latent = layers.Flatten()(mf_embedding_user(user_input))
    mf_item_latent = layers.Flatten()(mf_embedding_item(item_input))
    mf_vector = layers.multiply([mf_user_latent, mf_item_latent])

    # MLP part 
    mlp_user_latent = layers.Flatten()(mlp_embedding_user(user_input))
    mlp_item_latent = layers.Flatten()(mlp_embedding_item(item_input))
    mlp_vector = layers.concatenate([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = layers.Dense(layers_num[idx], kernel_regularizer=regularizers.l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    predict_vector = layers.concatenate([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = layers.Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_normal(),
                       name="prediction")(predict_vector)
    
    model_nuemf = models.Model(inputs=[user_input, item_input], outputs=prediction)
    model_nuemf.compile(optimizer=optimizers.Adam(lr=learning_rate, clipnorm=0.5), loss='binary_crossentropy')
    
    return model_nuemf