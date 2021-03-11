import time

import pandas as pd

from src import lstm_ns_random_model
from sklearn import cluster, metrics


def kmeans_cluster(num_clusters, word_embeddings):
    kmeans = cluster.KMeans(n_clusters=num_clusters, verbose=1)
    kmeans.fit(word_embeddings)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    silhouette_score = metrics.silhouette_score(word_embeddings, labels, metric='euclidean')
    print("Silhouette Score for the clusters = ", silhouette_score)

    return kmeans, labels, centroids


class NsClusteredModel(lstm_ns_random_model.NsRandomModel):
    def __init__(self, latent_dim, kc_dimension, input_file_path, gensim_model, one_hot_encoder):
        super().__init__(latent_dim, kc_dimension, input_file_path, gensim_model, one_hot_encoder)
        self.cluster_sampled_students = []
        self.cluster_sampled_problems = []

    def generate_training_sample(self, unique_students, unique_problems, n_std_clusters, n_prob_clusters):
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

        std_kmeans, std_cluster_labels, std_cluster_centers = kmeans_cluster(n_std_clusters, student_word_embeddings)
        prob_kmeans, prob_cluster_labels, prob_cluster_centers = kmeans_cluster(n_prob_clusters,
                                                                                problem_name_word_embeddings)

        for c in std_cluster_centers:
            std = self.word_embeddings.wv.most_similar(positive=[c], topn=1)[0][0]
            self.cluster_sampled_students.append(std)
        print("Sampled Students Size = ", len(self.cluster_sampled_students))

        for c in prob_cluster_centers:
            problem = self.word_embeddings.wv.most_similar(positive=[c], topn=1)[0][0]
            self.cluster_sampled_problems.append(problem)
        print("Sampled Problems Size = ", len(self.cluster_sampled_problems))

        file_iterator = pd.read_csv(self.input_file_path, sep="\t", header=0, iterator=True, chunksize=1000000)
        inputSequence = []
        outputSequence = []
        time_t1 = time.time()

        for chunk_data in file_iterator:
            for student_id, std_groups in chunk_data.groupby('Anon Student Id'):
                if student_id in self.cluster_sampled_students:
                    # time_it1 = time.time()
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
                                input_parameters = []
                                input_parameters.append(self.word_embeddings.wv[student_id])
                                input_parameters.append(self.word_embeddings.wv[prob_hierarchy_name])
                                input_parameters.append(self.word_embeddings.wv[problem_name])
                                for row in sub_skills:
                                    if str(row) != "nan":
                                        splitted_kcs = row.split("~~")
                                        for kc in splitted_kcs:
                                            kcs_prob.append(kc)
                                if kcs_prob:
                                    inputSequence.append(input_parameters)
                                    kcs_prob_one_hot = self.one_hot_encoder.transform_to_one_hot(kcs_prob)
                                    outputSequence.append(self.one_hot_encoder.append_start_and_end_tokens(kcs_prob_one_hot))
                    # time_it2 = time.time()
                    # print("Time Taken for ", problem_name, " = ", time_it2 - time_it1, " secs.")
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
