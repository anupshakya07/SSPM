# Student Strategy Prediction Using a Neuro-Symbolic Approach

## Abstract

Predicting student problem-solving strategies is a complex problem but one that can significantly impact automated instruction 
systems since they can adapt or personalize the system to suit the learner. While for small datasets, learning experts may 
be able to manually analyze data to infer student strategies, for large datasets, this approach is infeasible. We develop 
a Machine Learning model to predict strategies from student data with discrete interaction steps. Deep Neural Network (DNN) 
based methods such as LSTMs are a natural fit for this task since the goal is to model sequential data. However, purely 
LSTM-based methods often have long convergence times for large datasets and like several other DNN-based methods have the 
inherent problem of overfitting the data. To address these issues, we develop a Neuro-symbolic approach for strategy prediction, 
namely a model that combines strengths of symbolic AI (that can encode domain knowledge) with DNNs. Specifically, we encode 
relationships in the data using Markov Logic and use symmetries among these relationships to train an LSTM more efficiently. 
In particular, we use an importance sampling approach where we sample the training data such that for clusters/groups of symmetrical 
instances (instances where the strategies are likely to be symmetric), we only pick representative samples for training the model 
instead of using the whole group. Further, since some groups may contain more diverse strategies than the others, we adapt 
the importance weights based on previously observed samples. We run a detailed empirical evaluation on the publicly available
KDD EDM challenge datasets from Mathia where we show that by exploiting symmetries, we can learn a model that is both scalable 
and accurate.

## Using the code
This code implementation contains implementation of four major models: LSTM-NS-Random, LSTM-NS-NaiveGroup, LSTM-NS-Clustered 
and LSTM-NS-Adaptive.

#### LSTM-NS-Random
For this model, use the class named NsRandomModel. Initialize the constructor with the required parameters. Then, use the
generate_training_sample() method to generate samples for training. Then, use train_model() method to train the model. For
evaluating the model, call the following methods:
```
model.setup_inference_model()
model.evaluate_training_accuracy()

test_x, test_y, max_target_length = model.generate_sample(test_file_path,<num of samples>)
model.evaluate_model(test_x, test_y)
```

#### LSTM-NS-NaiveGroup
This model has the same usage as of the above model. For this model, use the class NsNaiveGroupModel.


#### LSTM-NS-Clustered
For this model, use the class NsClusteredModel. This model generates training samples by clustering the student and problem
groups separately. Provide the list of students and problems and the number of clusters for students and problems. Use 
generate_training_sample() to generate the samples. Then use train_model() to initiate training the model.

#### LSTM-NS-Adaptive
For this model, use the class NsAdaptiveModel. This model works in the following way:

```
Initialize with training samples with 100 student clusters and 1000 problem clusters
Generate the validation dataset for student clusters
Generate the validation dataset for problem clusters
for each iteration
    train the model
    calculate training accuracy
    calculate test accuracy
    evaluate performance on student clusters
    evaluate performance on problem clusters
    update the importance weights for each cluster
    sample new instances based on importance weights
```