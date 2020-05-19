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

#Server:distribute_task & evaluate
class Server:
    def __init__(self, epochs = 20000, verbose = 1,topK = 20, lr = 0.001,cilp_norm = 0.5, data_name = 'ml-1m', model_name = 'neumf'):
        self.epochs = epochs
        self.verbose = verbose
        self.topK = topK
        self.C = cilp_norm
        self.lr = lr
        #dataset
        t1 = time()
        dataset = Dataset("./Data/" + data_name)
        self.num_users, self.num_items = dataset.get_train_data_shape()
        self.test_datas = dataset.load_test_file()
        self.test_negatives = dataset.load_negative_file()
        print("Server Load data done [%.1f s]. #user=%d, #item=%d, #test=%d"
          % (time()-t1, self.num_users, self.num_items, len(self.test_datas)))
        #model
        if model_name == "gmf":
            self.model = get_compiled_gmf_model(self.num_users,self.num_items)
        elif model_name == "mlp":
            self.model = get_compiled_mlp_model(self.num_users,self.num_items)
        elif model_name == "neumf":
            self.model = get_compiled_neumf_model(self.num_users,self.num_items)
        #init clients
        self.client = Client()

    def distribute_task(self, client_ids):
        server_weights = self.model.get_weights()
        client_weight_datas = []
        for client_id in client_ids:
            weights = self.client.train_epoch(self.model, client_id, server_weights)
            client_weight_datas.append(weights)
        return client_weight_datas

    def federated_average(self, client_weight_datas):
        client_num = len(client_weight_datas)
        assert client_num != 0
        w = client_weight_datas[0]
        for i in range(1, client_num):
            w += client_weight_datas[i]
        w = w/client_num
        self.model.set_weights(w)
        return w

    def evaluate_model(self):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [[] for _ in range(self.topK)], [[] for _ in range(self.topK)]
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
                p = ranklist.index(gtItem)
                for i in range(p):
                    hits[i].append(0)
                    ndcgs[i].append(0)
                for i in range(p, self.topK):
                    hits[i].append(1)
                    ndcgs[i].append(math.log(2)/math.log(ranklist.index(gtItem)+2))
            else:
                for i in range(self.topK):
                    hits[i].append(0)
                    ndcgs[i].append(0)
        hits = [np.array(hits[i]).mean() for i in range(self.topK)]
        ndcgs = [np.array(ndcgs[i]).mean() for i in range(self.topK)]
        return hits, ndcgs


    def run(self):
        t1 = time()
        hrs, ndcgs = self.evaluate_model()
        for i in range(self.topK):
            print('HR@%d = %.4f, NDCG@%d = %.4f' % (i+1,hrs[i],i+1, ndcgs[i]))
        print('[%.1f s]' % (time()-t1))
        
        # Train model federated
#        best_hr, best_ndcg, best_iter = hr, ndcg, -1
        for epoch in range(self.epochs):
            t1 = time()
            for i in range(1000):
                server_weights = self.model.get_weights()
                client_weight_datas=self.distribute_task(random.sample(range(self.num_users),5))
                client_weights = self.federated_average(client_weight_datas)
                #self.C = 10*self.lr*max([np.linalg.norm(client_weights[la] - server_weights[la]) for la in range(len(client_weights))])
                #self.model.compile(optimizer=optimizers.Adam(lr=self.lr, clipnorm=self.C), loss='binary_crossentropy')
                #print(self.C)
                #self.model.set_weights(client_weights)
            t2 = time()
            print('Iteration %d [%.1f s]'
              % (epoch,  t2-t1))

            if epoch % self.verbose == 0:
                hrs, ndcgs = self.evaluate_model()
                for i in range(self.topK):
                    print('HR@%d = %.4f, NDCG@%d = %.4f' % (i+1,hrs[i],i+1, ndcgs[i]))
                print('[%.1f s]' % (time()-t1))
        '''
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
        '''

ser = Server(verbose = 20)
ser.run()