from time import time
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras import initializers,regularizers,optimizers
from Dataset import Dataset
from train_model import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Client:
    def __init__(self, batch_size = 64, epochs = 1,
        data_name = 'ml-1m', model_name = 'gmf'):
        self.epochs = epochs
        self.batch_size = batch_size
        #get dataset
        t1 = time()
        dataset = Dataset("./Data/ml-1m")
        self.num_users, self.num_items = dataset.get_train_data_shape()
        self.client_train_datas = dataset.load_client_train_date()
        print("Client Load data done [%.1f s]. #user=%d, #item=%d"
          % (time()-t1, self.num_users, self.num_items))


    def train_epoch(self,server_model, client_id, server_weights):
        train_data = self.client_train_datas[client_id]
        server_model.set_weights(server_weights)
        
        hist = server_model.fit([np.array(train_data[0]), np.array(train_data[1])],  # input
                         np.array(train_data[2]),  # labels
                         batch_size=self.batch_size, epochs=self.epochs, verbose=0, shuffle=True)
        '''
        for i in range(int(len(train_data[0])/self.batch_size)):
            gradients = None
            cur_weights = np.array(server_model.get_weights())
            begin = i * self.batch_size
            end = begin
            for j  in range(self.batch_size):
                if begin + j >= len(train_data[0]):
                    break;
                server_model.set_weights(cur_weights)
                server_model.fit([np.array(train_data[0][begin + j:begin + j + 1]), np.array(train_data[1][begin + j:begin + j + 1])],  # input
                        np.array(train_data[2][begin + j:begin + j + 1]),  # labels
                        batch_size=1, epochs=self.epochs, verbose=0, shuffle=True)
                end += 1
                if j != 0:
                    gradients += np.array(server_model.get_weights())
                else:
                    gradients = np.array(server_model.get_weights())
            server_model.set_weights(gradients/(end - begin))
        '''
        weights = np.array(server_model.get_weights())
        #add noise
        epsilon = 0.5
        delta = 0.00001
        sensitivity = 0.001/64 * math.sqrt(2 * math.log(1.25/delta))/epsilon
        sigma = sensitivity/epsilon * math.sqrt(2 * math.log(1.25/delta))
        #noise = np.random.normal(0, sigma)
        noise = np.random.normal(0, sigma/math.sqrt(5), weights.shape)
        '''
        #分层加入噪声
        for i in range(len(weights)):
            increment = weights[i] - server_weights[i]
            print("weights:")
            print(weights[i])
            print("delta:")
            print(increment)
            sensitivity = increment.max() - increment.min() 
            sigma = ss * sensitivity
            noise = np.random.normal(0, sigma, weights[i].shape)
            weights[i] += noise
        '''
        return weights

