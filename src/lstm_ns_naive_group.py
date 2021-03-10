import random
import time

import pandas as pd

from src import lstm_ns_random_model


class NsNaiveGroupModel(lstm_ns_random_model.NsRandomModel):
    DEFAULT_PROB_SAMPLE_SIZE = 10

    def __init__(self, latent_dim, kc_dimension, input_file_path, gensim_model, one_hot_encoder):
        super().__init__(latent_dim, kc_dimension, input_file_path, gensim_model, one_hot_encoder)

    def generate_training_sample(self, sample_size=10):
        self.DEFAULT_PROB_SAMPLE_SIZE = sample_size

        file_iterator = pd.read_csv(self.input_file_path, sep="\t", header=0, iterator=True, chunksize=1000000)

        inputSequence = []
        outputSequence = []
        time_t1 = time.time()
        for chunk_data in file_iterator:
            for student_id, std_groups in chunk_data.groupby('Anon Student Id'):
                if student_id in self.word_embeddings.wv.vocab:
                    student_problem_list = []
                    prob_name = std_groups.groupby('Problem Name')
                    for problem_name, prob_name_groups in prob_name:
                        student_problem_list.append(problem_name)
                    #             print("Student Problem Size -> ", len(student_problem_list))

                    sampled_problems = []
                    if len(student_problem_list) < self.DEFAULT_PROB_SAMPLE_SIZE:
                        sampled_problems = student_problem_list
                    else:
                        sampled_problems = random.sample(student_problem_list, self.DEFAULT_PROB_SAMPLE_SIZE)

                    for prob in sampled_problems:
                        if prob in self.word_embeddings.wv.vocab:
                            prob_group = prob_name.get_group(prob)
                            sub_skills = prob_group['KC(SubSkills)']
                            prob_hierarchy_name = ""
                            for p_hierarchy in prob_group['Problem Hierarchy']:
                                prob_hierarchy_name = p_hierarchy
                                break

                            kcs_prob = []
                            if prob_hierarchy_name in self.word_embeddings.wv.vocab:
                                input_parameters = []
                                input_parameters.append(self.word_embeddings.wv[student_id])
                                input_parameters.append(self.word_embeddings.wv[prob_hierarchy_name])
                                input_parameters.append(self.word_embeddings.wv[prob])
                                for row in sub_skills:
                                    if str(row) != "nan":
                                        splitted_kcs = row.split("~~")
                                        for kc in splitted_kcs:
                                            kcs_prob.append(kc)
                                if kcs_prob:
                                    inputSequence.append(input_parameters)
                                    kcs_prob_one_hot = self.one_hot_encoder.transform_to_one_hot(kcs_prob)
                                    outputSequence.append(self.one_hot_encoder.append_start_and_end_tokens(kcs_prob_one_hot))
        time_t2 = time.time()
        print("Time Taken = ", time_t2 - time_t1, " secs.")
        print("X Input List Length -> ", len(inputSequence))
        print("X Output List Length -> ", len(outputSequence))

        max_length_tar = 0
        for element in outputSequence:
            if len(element) > max_length_tar:
                max_length_tar = len(element)
        print("Max Output Sequence Length = ", max_length_tar)
        self.train_samples_x, self.train_samples_y, self.max_length_tar = inputSequence, outputSequence, max_length_tar

