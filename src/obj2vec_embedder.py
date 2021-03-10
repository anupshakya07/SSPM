import logging
import pandas as pd
import gensim.models
import time


class RelationWordEmbedding:
    def __init__(self, ip_path, embedding_dimension=300, sliding_window_size=8):
        self.input_file_path = ip_path
        self.embedding_dimension = embedding_dimension
        self.sliding_window_size = sliding_window_size
        self.sentences = []
        self.model = None

    def generate_sentences(self):
        chunk_iterator = pd.read_csv(self.input_file_path, sep="\t", header=0, iterator=True, chunksize=1000000)
        print("------Generating Sentences to train the word embeddings-------")
        start_time = time.time()

        for chunk_data in chunk_iterator:
            for student_id, std_groups in chunk_data.groupby('Anon Student Id'):
                prob_hierarchy = std_groups.groupby('Problem Hierarchy')
                for hierarchy, hierarchy_groups in prob_hierarchy:
                    prob_name = hierarchy_groups.groupby('Problem Name')
                    for problem_name, prob_name_groups in prob_name:
                        sub_skills = prob_name_groups['KC(SubSkills)']
                        for a in sub_skills:
                            if str(a) != "nan":
                                temp = a.split("~~")
                                self.sentences.append(temp)  # {KC1,KC2,....,KCn}
                                for kc in a.split("~~"):
                                    list_with_single_kc = [student_id, hierarchy, problem_name, kc]
                                    self.sentences.append(list_with_single_kc)  # {StdId, ProbHierarchy, ProblemName, KC}
                                    st_kc = [student_id, kc]
                                    self.sentences.append(st_kc)
                                    prob_kc = [problem_name, kc]
                                    self.sentences.append(prob_kc)
        end_time = time.time()
        print("Time Taken for generating the sentences = ", end_time - start_time)

    def train(self):
        self.generate_sentences()
        print("******************************* Training Relational Word Embeddings *******************************")
        print("Parameters:->>>>>>>")
        print("Number of sentences trained = ", len(self.sentences))
        print("Embedding Dimension of the Word Embeddings = ", self.embedding_dimension)
        print("Sliding Window Size = ", self.sliding_window_size)
        start_time = time.time()
        self.model = gensim.models.Word2Vec(self.sentences, size=self.embedding_dimension, window=self.sliding_window_size, min_count=1)
        end_time = time.time()
        print("Time taken to train the word embeddings = ", end_time - start_time, " secs.")

    def save_trained_model(self, model_name):
        self.model.save(model_name)

    def load_trained_model(self, model_name):
        self.model = gensim.models.Word2Vec.load(model_name)