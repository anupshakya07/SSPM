from src import model
import pandas as pd
import time


class NsRandomModel(model.LearningModel):
    def __init__(self, latent_dim, kc_dimension, input_file_path, gensim_model, one_hot_encoder):
        super().__init__(latent_dim, kc_dimension, input_file_path, gensim_model, one_hot_encoder)
        self.train_samples_x = []
        self.train_samples_y = []
        self.max_length_src = 3
        self.max_length_tar = None

    def generate_training_sample(self, n_rows):
        self.train_samples_x, self.train_samples_y, self.max_length_tar = self.generate_sample(self.input_file_path,
                                                                                               n_rows)

    def train_model(self, num_epochs, batch_size):
        self.create_model()
        print(self.model.summary())
        super().train_model(self.train_samples_x, self.train_samples_y, self.max_length_src, self.max_length_tar,
                            num_epochs, batch_size)

    def evaluate_training_accuracy(self, n):
        self.evaluate_model(self.train_samples_x[:n], self.train_samples_y[:n], self.max_length_src,
                            self.max_length_tar, self.one_hot_encoder.model)
