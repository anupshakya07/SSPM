from src import data_preprocessor
from src import one_hot_encoder
from src import obj2vec_embedder
from src import model, lstm_ns_random_model, lstm_ns_naive_group, lstm_ns_clustered, lstm_ns_adaptive

from os import path

if __name__ == "__main__":
    train_file_path = "datasets/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt"
    test_file_path = "datasets/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_test.txt"

    data_processor = data_preprocessor.DataPreprocessor(input_file_path=train_file_path)
    data_processor.analyze_dataset()

    obj_to_vec_embedder = obj2vec_embedder.RelationWordEmbedding(train_file_path, 300, 8)

    if path.exists("saved_models/test_kc_embedding"):
        obj_to_vec_embedder.load_trained_model(model_name="saved_models/test_kc_embedding")
    else:
        obj_to_vec_embedder.train()
        obj_to_vec_embedder.save_trained_model("saved_models/test_kc_embedding")

    one_hot_encoder = one_hot_encoder.OneHotEnc()
    one_hot_encoder.train(data_processor.unique_kcs)

    ################################################################################################
    # For LSTM-NS-Random Model
    # model = lstm_ns_random_model.NsRandomModel(200, len(data_processor.unique_kcs), train_file_path,
    #                                            obj_to_vec_embedder.model, one_hot_encoder)
    # model.generate_training_sample(n_rows=10000)
    # model.train_model(num_epochs=5, batch_size=50)
    ################################################################################################

    ################################################################################################
    # For LSTM-NS-NaiveGroup Model
    # model = lstm_ns_naive_group.NsNaiveGroupModel(200, len(data_processor.unique_kcs), train_file_path,
    #                                               obj_to_vec_embedder.model, one_hot_encoder)
    # model.generate_training_sample(sample_size=10)
    # model.train_model(num_epochs=5, batch_size=50)
    ################################################################################################

    ################################################################################################
    # For LSTM-NS-Clustered Model
    # model = lstm_ns_clustered.NsClusteredModel(200, len(data_processor.unique_kcs), train_file_path,
    #                                            obj_to_vec_embedder.model, one_hot_encoder)
    # model.generate_training_sample(data_processor.unique_students, data_processor.unique_problems, 20, 500)
    # model.train_model(num_epochs=5, batch_size=50)
    ################################################################################################

    ################################################################################################
    # For LSTM-NS-Clustered Model
    model = lstm_ns_adaptive.NsAdaptiveModel(200, len(data_processor.unique_kcs), train_file_path, test_file_path,
                                             obj_to_vec_embedder.model, one_hot_encoder)
    model.set_student_validation_size(size=3)
    model.set_problem_validation_size(size=3)
    model.generate_training_sample(data_processor.unique_students, data_processor.unique_problems, 100, 100)
    model.train_model(num_epochs=30, batch_size=100, sampling_factor=4, n_iter=10)
    ################################################################################################

    #### Save the Trained Model ####
    # model.save_model("test_model")

    #### Training Accuracy and Test Accuracy for LSTM-NS-Random, LSTM-NS-NaiveGroup and LSTM-NS-Clustered models ####
    # model.setup_inference_model()
    # model.evaluate_training_accuracy(100)

    # test_x, test_y, max_target_length = model.generate_sample(test_file_path, 200)
    # model.evaluate_model(test_x, test_y, 3, max_target_length, one_hot_encoder.model, "Test")
