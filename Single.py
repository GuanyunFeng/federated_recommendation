from time import time
import os
import math
import random
import heapq
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras import initializers,regularizers,optimizers
from Dataset import Dataset
from Client import Client
from train_model import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Single():
    def __init__(self, epochs = 100, verbose = 5,topK = 10,data_name = 'ml-1m', model_name = 'gmf'):
        self.epochs = epochs
        self.verbose = verbose
        self.topK = topK
        #dataset
        t1 = time()
        dataset = Dataset("./Data/" + data_name)
        self.num_users, self.num_items = dataset.get_train_data_shape()
        self.test_datas = dataset.load_test_file()
        self.test_negatives = dataset.load_negative_file()
        self.train_datas = dataset.load_train_file()
        print("Server Load data done [%.1f s]. #user=%d, #item=%d, #test=%d"
          % (time()-t1, self.num_users, self.num_items, len(self.test_datas)))
        #model
        if model_name == "gmf":
            self.model = get_compiled_gmf_model(self.num_users,self.num_items)
        elif model_name == "mlp":
            self.model = get_compiled_mlp_model(self.num_users,self.num_items)
        elif model_name == "neumf":
            self.model = get_compiled_neumf_model(self.num_users,self.num_items)
    
    def evaluate_model(self):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        for idx in range(len(self.test_datas)):
            rating = self.test_datas[idx]
            items = self.test_negatives[idx]
            user_id = rating[0]
            gtItem = rating[1]
            items.append(gtItem)
            # Get prediction scores
            map_item_score = {}
            users = np.full(len(items), user_id, dtype='int32')
            predictions = self.model.predict([users, np.array(items)],
                                        batch_size=100, verbose=0)
            for i in range(len(items)):
                item = items[i]
                map_item_score[item] = predictions[i]
            items.pop()
            # Evaluate top rank list
            ranklist = heapq.nlargest(self.topK, map_item_score, key=map_item_score.get)
            if gtItem in ranklist:
                hits.append(1)
                ndcgs.append(math.log(2)/math.log(ranklist.index(gtItem)+2))
            else:
                hits.append(0)
                ndcgs.append(0)
        return np.array(hits).mean(), np.array(ndcgs).mean()


    def run(self):
        t1 = time()
        hr, ndcg = self.evaluate_model()
        print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
        
        # Train model federated
        best_hr, best_ndcg, best_iter = hr, ndcg, -1
        for epoch in range(self.epochs):
            t1 = time()
            hist = self.model.fit([np.array(self.train_datas[0]), np.array(self.train_datas[1])],  # input
                         np.array(self.train_datas[2]),  # labels
                         batch_size=256, epochs=1, verbose=0, shuffle=True)
            t2 = time()
            print('Iteration %d [%.1f s]'
              % (epoch,  t2-t1))

            if epoch % self.verbose == 0:
                hr, ndcg = self.evaluate_model()
                print('HR = %.4f, NDCG = %.4f [%.1f s]'
                  % (hr, ndcg, time()-t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))

S = Single(model_name="neumf")
S.run()