import scipy.sparse as sp
import numpy as np
import collections

class Dataset(object):

    def __init__(self, path):
        self.path = path
        self.num_users, self.num_items = self.get_train_data_shape()




    def get_train_data_shape(self):
        filename = self.path + ".train.rating"
        num_users=0
        num_items = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        num_items += 1
        num_users += 1
        return num_users, num_items




    def load_test_file(self):
        filename = self.path + ".test.rating"
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList




    def load_negative_file(self):
        filename = self.path + ".test.negative"
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList




    def load_train_file(self):
        filename = self.path + ".train.rating"
        num_users, num_items = self.get_train_data_shape()

        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                #print("usr:{} item:{} score:{}".format(user,item,rating))
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()

        train_datas=[[],[],[]]
        with open(filename, "r") as f:
            for (usr, item) in mat.keys():
                train_datas[0].append(usr)
                train_datas[1].append(item)
                train_datas[2].append(1)
                line = f.readline()
                for t in range(4):
                    nega_item = np.random.randint(num_items)
                    while (usr, nega_item) in mat.keys():
                        nega_item = np.random.randint(num_items)
                train_datas[0].append(usr)
                train_datas[1].append(nega_item)
                train_datas[2].append(0)
        return train_datas




    def load_client_train_date(self):
        filename = self.path + ".train.rating"
        num_users, num_items = self.get_train_data_shape()

        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                #print("usr:{} item:{} score:{}".format(user,item,rating))
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()

        client_datas=[[[],[],[]] for i in range(num_users)]
        with open(filename, "r") as f:
            for (usr, item) in mat.keys():
                client_datas[usr][0].append(usr)
                client_datas[usr][1].append(item)
                client_datas[usr][2].append(1)
                line = f.readline()
                for t in range(4):
                    nega_item = np.random.randint(num_items)
                    while (usr, nega_item) in mat.keys():
                        nega_item = np.random.randint(num_items)
                client_datas[usr][0].append(usr)
                client_datas[usr][1].append(nega_item)
                client_datas[usr][2].append(0)
        return client_datas