from src import model
import pandas as pd
import time


class NsRandomModel(model.LearningModel):
    def __init__(self, latent_dim, kc_dimension, input_file_path, gensim_model, one_hot_encoder):
        super().__init__(latent_dim, kc_dimension)
        self.input_file_path = input_file_path
        self.word_embeddings = gensim_model
        self.one_hot_encoder = one_hot_encoder
        self.random_samples_x = []
        self.random_samples_y = []
        self.max_length_src = 3
        self.max_length_tar = None

    def generate_training_sample(self, n_rows):
        self.random_samples_x, self.random_samples_y, self.max_length_tar = self.generate_sample(self.input_file_path,
                                                                                                 n_rows)

    def generate_sample(self, file_path, n_rows):
        start_time = time.time()
        data_sample = pd.read_csv(file_path, sep="\t", header=0, nrows=n_rows)
        samples_x = []
        samples_y = []

        for student_id, std_groups in data_sample.groupby('Anon Student Id'):
            if student_id in self.word_embeddings.wv.vocab:
                problem_hierarchy_groups = std_groups.groupby('Problem Hierarchy')
                for ph_name, ph_groups in problem_hierarchy_groups:
                    if ph_name in self.word_embeddings.wv.vocab:
                        problem_groups = ph_groups.groupby('Problem Name')
                        for prob_name, prob_group in problem_groups:
                            if prob_name in self.word_embeddings.wv.vocab:
                                sub_skills = prob_group['KC(SubSkills)']
                                kcs_prob = []
                                input_parameters = [self.word_embeddings.wv[student_id],
                                                    self.word_embeddings.wv[ph_name],
                                                    self.word_embeddings.wv[prob_name]]

                                for row in sub_skills:
                                    if str(row) != "nan":
                                        split_kcs = row.split("~~")
                                        for kc in split_kcs:
                                            kcs_prob.append(kc)
                                if kcs_prob:
                                    samples_x.append(input_parameters)
                                    kcs_prob_one_hot = self.one_hot_encoder.transform_to_one_hot(kcs_prob)
                                    samples_y.append(self.one_hot_encoder.append_start_and_end_tokens(kcs_prob_one_hot))
        end_time = time.time()
        print("Time Taken = ", end_time - start_time)
        print("X Input List Length -> ", len(samples_x))
        print("X Output List Length -> ", len(samples_y))

        max_length_tar = 0
        for element in samples_y:
            if len(element) > max_length_tar:
                max_length_tar = len(element)
        print("Max Output Sequence Length = ", max_length_tar)
        return samples_x, samples_y, max_length_tar

    def train_model(self, num_epochs, batch_size):
        self.create_model()
        print(self.model.summary())
        super().train_model(self.random_samples_x, self.random_samples_y, self.max_length_src, self.max_length_tar,
                            num_epochs, batch_size)

    def evaluate_training_accuracy(self, n):
        self.evaluate_model(self.random_samples_x[:n], self.random_samples_y[:n], self.max_length_src,
                            self.max_length_tar, self.one_hot_encoder.model)
