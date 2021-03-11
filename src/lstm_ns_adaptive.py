from sklearn.model_selection import train_test_split

from src import lstm_ns_clustered, model
from sklearn import cluster

import time, random
import pandas as pd
import numpy as np


class NsAdaptiveModel(lstm_ns_clustered.NsClusteredModel):
    STUDENT_VALIDATION_SIZE = 3
    PROBLEM_VALIDATION_SIZE = 3

    def __init__(self, latent_dim, kc_dimension, input_file_path, test_file_path, gensim_model, one_hot_encoder):
        super().__init__(latent_dim, kc_dimension, input_file_path, gensim_model, one_hot_encoder)
        self.test_file_path = test_file_path
        self.student_cluster_validation_data = {}
        self.problem_cluster_validation_data = {}
        self.student_cluster_size = None
        self.problem_cluster_size = None
        self.student_cluster_centers = []
        self.problem_cluster_centers = []
        self.student_cluster_map = {}
        self.problem_cluster_map = {}

    def set_student_validation_size(self, size):
        self.STUDENT_VALIDATION_SIZE = size

    def set_problem_validation_size(self, size):
        self.PROBLEM_VALIDATION_SIZE = size

    def generate_training_sample(self, unique_students, unique_problems, n_std_clusters, n_prob_clusters):
        self.student_cluster_size = n_std_clusters
        self.problem_cluster_size = n_prob_clusters
        self.generate_initial_sample(unique_students, unique_problems, n_std_clusters, n_prob_clusters)
        self.generate_student_validation_data()
        self.generate_problem_validation_data()

    def train_model(self, num_epochs, batch_size, sampling_factor, n_iter):

        self.create_model()

        student_importance_weights = np.ones(shape=(self.student_cluster_size), dtype=np.int8)
        problem_importance_weights = np.ones(shape=(self.problem_cluster_size), dtype=np.int8)

        x_train, x_test, y_train, y_test = train_test_split(self.train_samples_x, self.train_samples_y,
                                                            test_size=self.TEST_SIZE)
        test_sample_x, test_sample_y, test_max_length_tar = self.generate_sample(self.test_file_path, 10000)

        for iter_index in range(n_iter):
            time_t1 = time.time()
            self.train(x_train, x_test, y_train, y_test, self.max_length_src, self.max_length_tar, num_epochs,
                       batch_size)
            self.setup_inference_model()

            self.evaluate_model(x_train[:100], y_train[:100], self.max_length_src, self.max_length_tar,
                                self.one_hot_encoder.model, "Training Iteration " + str(iter_index+1) + " ")

            self.evaluate_model(test_sample_x, test_sample_y, 3, test_max_length_tar, self.one_hot_encoder.model,
                                "Test Iteration " + str(iter_index+1) + " ")

            student_validation_accuracy = []
            for index, studentID in enumerate(self.student_cluster_centers):
                if studentID in self.student_cluster_validation_data.keys():
                    student_input_sequence = self.student_cluster_validation_data.get(studentID)[0]
                    student_output_sequence = self.student_cluster_validation_data.get(studentID)[1]

                    accuracy = self.evaluate_model(student_input_sequence, student_output_sequence, self.max_length_src
                                                   , self.max_length_tar, self.one_hot_encoder.model
                                                   , "Student Validation ")
                    student_validation_accuracy.append(accuracy)

                    if accuracy < 0.5:
                        student_importance_weights[index] = student_importance_weights[index] + 1
                    else:
                        print("Okay for ", studentID)

                else:
                    print("This student is not in the cluster validation data keys -> ", studentID)
            np_student_validation_accuracy = np.array(student_validation_accuracy)
            print("STUDENT VALIDATION ACCURACY ->>> ", np_student_validation_accuracy.mean())

            problem_validation_accuracy = []
            for index, problemName in enumerate(self.problem_cluster_centers):
                if problemName in self.problem_cluster_validation_data.keys():
                    problem_input_sequence = self.problem_cluster_validation_data.get(problemName)[0]
                    problem_output_sequence = self.problem_cluster_validation_data.get(problemName)[1]

                    accuracy = self.evaluate_model(problem_input_sequence, problem_output_sequence, self.max_length_src,
                                                   self.max_length_tar, self.one_hot_encoder.model,
                                                   "Problem Validation ")
                    problem_validation_accuracy.append(accuracy)

                    if accuracy < 0.5:
                        problem_importance_weights[index] = problem_importance_weights[index] + 1
                    else:
                        print("Okay for ", problemName)
                else:
                    print("This problem is not in the cluster validation data keys -> ", problemName)
            np_problem_validation_accuracy = np.array(problem_validation_accuracy)
            print("PROBLEM VALIDATION ACCURACY ->>> ", np_problem_validation_accuracy.mean())

            file_iterator = pd.read_csv(self.input_file_path, sep="\t", header=0, iterator=True, chunksize=1000000)
            sampled_input_sequence = []
            sampled_output_sequence = []

            new_students = []
            new_problems = []

            for student_index, std in enumerate(self.student_cluster_centers):
                student_list = self.student_cluster_map.get(std)
                list_for_clustering = list(set(student_list) - set(self.cluster_sampled_students))

                embeddings = []

                num_samples = int((student_importance_weights[student_index] * self.student_cluster_size
                                   * sampling_factor) / student_importance_weights.sum())
                #         print("Num samples for student cluster ", std, " = ", num_samples)
                if num_samples > 0:
                    sampled_stds = []
                    if len(list_for_clustering) > num_samples:
                        student_kmeans = cluster.KMeans(n_clusters=num_samples, verbose=0)
                        for student in list_for_clustering:
                            if student in self.word_embeddings.wv.vocab:
                                embeddings.append(self.word_embeddings.wv[student])

                        student_kmeans.fit(embeddings)
                        sampled_std_embeddings = student_kmeans.cluster_centers_

                        for emb in sampled_std_embeddings:
                            max_similarity = 0
                            most_similar_std = ""
                            for std_id in list_for_clustering:
                                if std_id in self.word_embeddings.wv.vocab:
                                    sim = model.cosine_similarity(emb, self.word_embeddings.wv[std_id])
                                    if sim > max_similarity:
                                        max_similarity = sim
                                        most_similar_std = std_id
                            sampled_stds.append(most_similar_std)
                    else:
                        sampled_stds = list_for_clustering

                    new_students.extend(sampled_stds)

            for problem_name_index, prblm in enumerate(self.problem_cluster_centers):
                problem_list = self.problem_cluster_map.get(prblm)
                list_for_clustering = list(set(problem_list) - set(self.cluster_sampled_problems))

                embeddings = []

                num_samples = int((problem_importance_weights[problem_name_index] * self.problem_cluster_size *
                                   sampling_factor) / problem_importance_weights.sum())

                if num_samples > 0:
                    sampled_prbs = []
                    if len(list_for_clustering) > num_samples:
                        problem_kmeans = cluster.KMeans(n_clusters=num_samples, verbose=0)
                        for p in list_for_clustering:
                            if p in self.word_embeddings.wv.vocab:
                                embeddings.append(self.word_embeddings.wv[p])

                        problem_kmeans.fit(embeddings)
                        sampled_problem_embeddings = problem_kmeans.cluster_centers_

                        for emb in sampled_problem_embeddings:
                            max_similarity = 0
                            most_similar_problem = ""
                            for prb_name in list_for_clustering:
                                if prb_name in self.word_embeddings.wv.vocab:
                                    sim = model.cosine_similarity(emb, self.word_embeddings.wv[prb_name])
                                    if sim > max_similarity:
                                        max_similarity = sim
                                        most_similar_problem = prb_name
                            sampled_prbs.append(most_similar_problem)
                    else:
                        sampled_prbs = list_for_clustering

                    new_problems.extend(sampled_prbs)

            for chunk_data in file_iterator:
                for student_id, std_groups in chunk_data.groupby('Anon Student Id'):
                    if student_id in new_students:
                        problem_groups = std_groups.groupby('Problem Name')
                        for problem_name, prob_grp in problem_groups:
                            if problem_name in new_problems:
                                sub_skills = prob_grp['KC(SubSkills)']
                                prob_hierarchy_name = ""
                                for p_hierarchy in prob_grp['Problem Hierarchy']:
                                    prob_hierarchy_name = p_hierarchy
                                    break

                                kcs_prob = []
                                if prob_hierarchy_name in self.word_embeddings.wv.vocab:
                                    input_parameters = [self.word_embeddings.wv[student_id],
                                                        self.word_embeddings.wv[prob_hierarchy_name],
                                                        self.word_embeddings.wv[problem_name]]
                                    for row in sub_skills:
                                        if str(row) != "nan":
                                            splitted_kcs = row.split("~~")
                                            for kc in splitted_kcs:
                                                kcs_prob.append(kc)
                                    if kcs_prob:
                                        sampled_input_sequence.append(input_parameters)
                                        kcs_prob_one_hot = self.one_hot_encoder.transform_to_one_hot(kcs_prob)
                                        sampled_output_sequence.append(self.one_hot_encoder.append_start_and_end_tokens(kcs_prob_one_hot))
            time_t2 = time.time()
            print("Time Taken = ", time_t2 - time_t1)
            print("Size of collected samples -> ", len(sampled_input_sequence))

            self.cluster_sampled_students.extend(new_students)
            self.cluster_sampled_problems.extend(new_problems)

            x_train, x_test, y_train, y_test = train_test_split(sampled_input_sequence, sampled_output_sequence,
                                                                test_size=self.TEST_SIZE)
            for element in sampled_output_sequence:
                if len(element) > self.max_length_tar:
                    self.max_length_tar = len(element)
            print("Max Output Sequence Length = ", self.max_length_tar)

    def train(self, x_train, x_test, y_train, y_test , max_length_src, max_length_tar, num_epochs=30,
              batch_size=100):

        num_train_samples = len(x_train)
        num_val_samples = len(x_test)

        print("************************** Model Train Initiated ****************************")
        print("Parameters:")
        print("Number of Train instances = ", num_train_samples)
        print("Number of Validation instances = ", num_val_samples)
        print("Batch Size = ", batch_size)
        print("Number of Epochs = ", num_epochs)
        print("Learning Rate = ", self.learning_rate)
        print("Dropout Rate = ", self.dropout_rate)
        print("Latent Dimension = ", self.latent_dim)
        print("*****************************************************************************")

        history = self.model.fit_generator(
            generator=self.generate_batch(x_train, y_train, batch_size=batch_size,
                                          input_dimension=self.EMBEDDING_DIMENSION,
                                          output_dimension=self.kc_dimension, max_length_src=max_length_src,
                                          max_length_tar=max_length_tar),
            steps_per_epoch=num_train_samples // batch_size, epochs=num_epochs,
            validation_data=self.generate_batch(x_test, y_test, batch_size=batch_size,
                                                input_dimension=self.EMBEDDING_DIMENSION,
                                                output_dimension=self.kc_dimension, max_length_src=max_length_src,
                                                max_length_tar=max_length_tar),
            validation_steps=num_val_samples // batch_size, verbose=1)

        return history

    def generate_student_validation_data(self):
        chunk_iterator = pd.read_csv(self.input_file_path, sep="\t", header=0, iterator=True, chunksize=1000000)
        self.student_cluster_validation_data = {}
        time_t1 = time.time()

        for chunk_data in chunk_iterator:
            for student_id, std_groups in chunk_data.groupby('Anon Student Id'):
                if student_id in self.cluster_sampled_students:
                    time_it1 = time.time()
                    problem_groups = std_groups.groupby('Problem Name')
                    all_std_problems = []
                    for problem_name, prob_group in problem_groups:
                        if problem_name in self.word_embeddings.wv.vocab and problem_name not in self.cluster_sampled_problems:
                            all_std_problems.append(problem_name)
                    random_picked_problems = []
                    # print("Student ", student_id, " -> other problems size -> ", len(all_std_problems))
                    if len(all_std_problems) < self.STUDENT_VALIDATION_SIZE:
                        random_picked_problems = all_std_problems
                    else:
                        random_picked_problems = random.sample(all_std_problems, self.STUDENT_VALIDATION_SIZE)

                    student_validation_input_sequence = []
                    student_validation_output_sequence = []

                    for picked_problem in random_picked_problems:
                        problm_grp = problem_groups.get_group(picked_problem)
                        sub_skills = problm_grp['KC(SubSkills)']
                        probHierarchyName = ""
                        for p_hierarchy in problm_grp['Problem Hierarchy']:
                            probHierarchyName = p_hierarchy
                            break

                        kcs_prob = []
                        if probHierarchyName in self.word_embeddings.wv.vocab:
                            input_parameters = []
                            input_parameters.append(self.word_embeddings.wv[student_id])
                            input_parameters.append(self.word_embeddings.wv[probHierarchyName])
                            input_parameters.append(self.word_embeddings.wv[picked_problem])
                            for row in sub_skills:
                                if (str(row) != "nan"):
                                    splitted_kcs = row.split("~~")
                                    for kc in splitted_kcs:
                                        kcs_prob.append(kc)
                            if kcs_prob:
                                student_validation_input_sequence.append(input_parameters)
                                kcs_prob_one_hot = self.one_hot_encoder.transform_to_one_hot(kcs_prob)
                                student_validation_output_sequence.append(
                                    self.one_hot_encoder.append_start_and_end_tokens(kcs_prob_one_hot))

                    if student_validation_input_sequence:
                        self.student_cluster_validation_data.update(
                            {student_id: [student_validation_input_sequence, student_validation_output_sequence]})

                    time_it2 = time.time()
                    print("Time Taken for ", student_id, " = ", time_it2 - time_it1, " secs.")

                    for element in student_validation_output_sequence:
                        if len(element) > self.max_length_tar:
                            self.max_length_tar = len(element)
                    print("Max Output Sequence Length = ", self.max_length_tar)
        time_t2 = time.time()
        print("Total Time Taken to generate student validation data = ", time_t2 - time_t1, " secs.")

    def generate_problem_validation_data(self):
        data = pd.read_csv(self.input_file_path, sep="\t")

        self.problem_cluster_validation_data = {}
        time_t1 = time.time()

        for problem_name, problem_groups in data.groupby("Problem Name"):
            if problem_name in self.cluster_sampled_problems:
                time_it1 = time.time()
                student_groups = problem_groups.groupby('Anon Student Id')

                all_prob_students = []
                for student_name, std_group in student_groups:
                    if student_name not in self.cluster_sampled_students:
                        all_prob_students.append(student_name)

                random_picked_students = []
                print("Problem ", problem_name, " -> other students size -> ", len(all_prob_students))
                if len(all_prob_students) < self.PROBLEM_VALIDATION_SIZE:
                    random_picked_students = all_prob_students
                else:
                    random_picked_students = random.sample(all_prob_students, self.PROBLEM_VALIDATION_SIZE)

                problem_validation_input_sequence = []
                problem_validation_output_sequence = []

                for picked_student in random_picked_students:
                    student_grp = student_groups.get_group(picked_student)
                    sub_skills = student_grp['KC(SubSkills)']
                    probHierarchyName = ""
                    for p_hierarchy in student_grp['Problem Hierarchy']:
                        probHierarchyName = p_hierarchy
                        break

                    kcs_prob = []
                    if probHierarchyName in self.word_embeddings.wv.vocab:
                        input_parameters = []
                        input_parameters.append(self.word_embeddings.wv[picked_student])
                        input_parameters.append(self.word_embeddings.wv[probHierarchyName])
                        input_parameters.append(self.word_embeddings.wv[problem_name])
                        for row in sub_skills:
                            if str(row) != "nan":
                                splitted_kcs = row.split("~~")
                                for kc in splitted_kcs:
                                    kcs_prob.append(kc)
                        if kcs_prob:
                            problem_validation_input_sequence.append(input_parameters)
                            kcs_prob_one_hot = self.one_hot_encoder.transform_to_one_hot(kcs_prob)
                            problem_validation_output_sequence.append(
                                self.one_hot_encoder.append_start_and_end_tokens(kcs_prob_one_hot))
                if problem_validation_input_sequence:
                    self.problem_cluster_validation_data.update(
                        {problem_name: [problem_validation_input_sequence, problem_validation_output_sequence]})
                time_it2 = time.time()
                print("Time Taken for ", problem_name, " = ", time_it2 - time_it1, " secs.")

                for element in problem_validation_output_sequence:
                    if len(element) > self.max_length_tar:
                        self.max_length_tar = len(element)
                print("Max Output Sequence Length = ", self.max_length_tar)
        time_t2 = time.time()
        print("Total Time Taken to generate problem validation data = ", time_t2 - time_t1, " secs.")

    def generate_initial_sample(self, unique_students, unique_problems, n_std_clusters, n_prob_clusters):
        student_word_embeddings = []
        for index, student_id in enumerate(unique_students):
            if student_id in self.word_embeddings.wv.vocab:
                student_word_embeddings.append(self.word_embeddings.wv[student_id])
        print("Student Word Embedding List Length -> ", len(student_word_embeddings))

        problem_name_word_embeddings = []
        for prob in unique_problems:
            if prob in self.word_embeddings.wv.vocab:
                problem_name_word_embeddings.append(self.word_embeddings.wv[prob])
        print("Problem Name Word Embeddings Length = ", len(problem_name_word_embeddings))

        std_kmeans, std_cluster_labels, std_cluster_centers = lstm_ns_clustered.kmeans_cluster(n_std_clusters,
                                                                                               student_word_embeddings)
        prob_kmeans, prob_cluster_labels, prob_cluster_centers = lstm_ns_clustered.kmeans_cluster(n_prob_clusters,
                                                                                problem_name_word_embeddings)

        self.student_cluster_centers = []
        self.student_cluster_map = {}
        for c in std_cluster_centers:
            std = self.word_embeddings.wv.most_similar(positive=[c], topn=1)[0][0]
            self.cluster_sampled_students.append(std)
            self.student_cluster_centers.append(std)
            centroid_cluster_label = std_kmeans.predict([c])[0]
            student_list = []
            for i, label in enumerate(std_cluster_labels):
                if label == centroid_cluster_label:
                    std_emb = student_word_embeddings[i]
                    most_similar_std = self.word_embeddings.wv.most_similar(positive =[std_emb], topn=1)[0][0]
                    student_list.append(most_similar_std)
            self.student_cluster_map.update({std:student_list})
        print("Sampled Students Size = ", len(self.cluster_sampled_students))

        self.problem_cluster_centers = []
        self.problem_cluster_map = {}
        for c in prob_cluster_centers:
            problem = self.word_embeddings.wv.most_similar(positive=[c], topn=1)[0][0]
            self.cluster_sampled_problems.append(problem)
            self.problem_cluster_centers.append(problem)
            centroid_cluster_label = prob_kmeans.predict([c])[0]
            problem_list = []

            for i, label in enumerate(prob_cluster_labels):
                if label == centroid_cluster_label:
                    prob_emb = problem_name_word_embeddings[i]
                    most_similar_prob = self.word_embeddings.wv.most_similar(positive=[prob_emb], topn=1)[0][0]
                    problem_list.append(most_similar_prob)
            self.problem_cluster_map.update({problem: problem_list})
        print("Sampled Problems Size = ", len(self.cluster_sampled_problems))

        file_iterator = pd.read_csv(self.input_file_path, sep="\t", header=0, iterator=True, chunksize=1000000)
        inputSequence = []
        outputSequence = []
        time_t1 = time.time()

        for chunk_data in file_iterator:
            for student_id, std_groups in chunk_data.groupby('Anon Student Id'):
                if student_id in self.cluster_sampled_students:
                    problem_groups = std_groups.groupby('Problem Name')
                    for problem_name, prob_grp in problem_groups:
                        if problem_name in self.cluster_sampled_problems:
                            sub_skills = prob_grp['KC(SubSkills)']
                            prob_hierarchy_name = ""
                            for p_hierarchy in prob_grp['Problem Hierarchy']:
                                prob_hierarchy_name = p_hierarchy
                                break

                            kcs_prob = []
                            if prob_hierarchy_name in self.word_embeddings.wv.vocab:
                                input_parameters = [self.word_embeddings.wv[student_id],
                                                    self.word_embeddings.wv[prob_hierarchy_name],
                                                    self.word_embeddings.wv[problem_name]]
                                for row in sub_skills:
                                    if str(row) != "nan":
                                        splitted_kcs = row.split("~~")
                                        for kc in splitted_kcs:
                                            kcs_prob.append(kc)
                                if kcs_prob:
                                    inputSequence.append(input_parameters)
                                    kcs_prob_one_hot = self.one_hot_encoder.transform_to_one_hot(kcs_prob)
                                    outputSequence.append(
                                        self.one_hot_encoder.append_start_and_end_tokens(kcs_prob_one_hot))
        time_t2 = time.time()
        print("Time Taken = ", time_t2 - time_t1, " secs")
        print("X Input List Length -> ", len(inputSequence))
        print("X Output List Length -> ", len(outputSequence))

        max_length_tar = 0
        for element in outputSequence:
            if len(element) > max_length_tar:
                max_length_tar = len(element)
        print("Max Output Sequence Length = ", max_length_tar)

        self.train_samples_x, self.train_samples_y, self.max_length_tar = inputSequence, outputSequence, max_length_tar

