#
#
#  Base Class for pre-processing the dataset
#
#  Contains built-in functions to analyze the datsets from KDD Cup Challenge
#
#  Developed by : Anup Shakya (anupshakya07@gmail.com)

import time
import pandas as pd


class DataPreprocessor:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        self.unique_students = None
        self.unique_problems = None
        self.unique_prob_hierarchy = None
        self.unique_kcs = None

    def analyze_dataset(self):
        file_iterator = self.load_file_iterator()

        start_time = time.time()
        self.unique_students = {"st"}
        self.unique_problems = {"pr"}
        self.unique_prob_hierarchy = {"ph"}
        self.unique_kcs = {"kc"}
        for chunk_data in file_iterator:
            for student_id, std_groups in chunk_data.groupby('Anon Student Id'):
                self.unique_students.update({student_id})
                prob_hierarchy = std_groups.groupby('Problem Hierarchy')
                for hierarchy, hierarchy_groups in prob_hierarchy:
                    self.unique_prob_hierarchy.update({hierarchy})
                    prob_name = hierarchy_groups.groupby('Problem Name')
                    for problem_name, prob_name_groups in prob_name:
                        self.unique_problems.update({problem_name})
                        sub_skills = prob_name_groups['KC(SubSkills)']
                        for a in sub_skills:
                            if str(a) != "nan":
                                temp = a.split("~~")
                                for kc in temp:
                                    self.unique_kcs.update({kc})
        self.unique_students.remove("st")
        self.unique_problems.remove("pr")
        self.unique_prob_hierarchy.remove("ph")
        self.unique_kcs.remove("kc")
        end_time = time.time()
        print("Time Taken to analyze dataset = ", end_time - start_time)
        print("Length of unique students->", len(self.unique_students))
        print("Length of unique problems->", len(self.unique_problems))
        print("Length of unique problem hierarchy->", len(self.unique_prob_hierarchy))
        print("Length of Unique Knowledge components ->", len(self.unique_kcs))

    def load_file_iterator(self):
        chunk_iterator = pd.read_csv(self.input_file_path, sep="\t", header=0, iterator=True, chunksize=1000000)
        return chunk_iterator
