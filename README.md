# Prediction Models


## Using the code
This code implementation contains implementation of four major models: LSTM-NS-Random, LSTM-NS-NaiveGroup, LSTM-NS-Clustered 
and LSTM-NS-Adaptive.

## Compiler and Package Requirements
Python 3.6
Gensim 3.x
Tensorflow 2.x


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
