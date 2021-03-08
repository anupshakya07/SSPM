from sklearn.preprocessing import OneHotEncoder
import numpy as np


class OneHotEnc:
    def __init__(self):
        self.model = OneHotEncoder(sparse=False)
        self.kc_dimension = None

    def train(self, list_words):
        np_array = np.array(list(list_words)).reshape(-1, 1)
        self.model.fit(np_array)
        self.kc_dimension = len(list_words)

    def transform_to_one_hot(self, kc_seq):
        kc_2d = np.array(kc_seq).reshape(-1, 1)
        one_hot = self.model.transform(kc_2d)
        return one_hot

    def transform_one_hot_to_kcs(self, one_hot_list):
        kcs_list = self.model.inverse_transform(one_hot_list)
        return kcs_list

    def append_start_and_end_tokens(self, arr):
        start_token = np.ones(self.kc_dimension)
        end_token = np.ones(self.kc_dimension)
        arr_list = arr.tolist()
        arr_list.insert(0, start_token)
        arr_list.append(end_token)
        return np.array(arr_list)
