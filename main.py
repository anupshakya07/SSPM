from src import data_preprocessor
from src import one_hot_encoder
from src import obj2vec_embedder
from src import lstm_ns_random_model
from src import model

if __name__ == "__main__":
    train_file_path = "datasets/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt"
    test_file_path = "datasets/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_test.txt"

    data_processor = data_preprocessor.DataPreprocessor(input_file_path=train_file_path)
    data_processor.analyze_dataset()

    obj_to_vec_embedder = obj2vec_embedder.RelationWordEmbedding(train_file_path, 300, 8)
    obj_to_vec_embedder.train()
    obj_to_vec_embedder.save_trained_model("saved_models/test_kc_embedding")

    one_hot_encoder = one_hot_encoder.OneHotEnc()
    one_hot_encoder.train(data_processor.unique_kcs)

    model = lstm_ns_random_model.NsRandomModel(200, len(data_processor.unique_kcs), train_file_path,
                                               obj_to_vec_embedder.model, one_hot_encoder)
    model.generate_training_sample(10000)
    model.train_model(5, 50)
    # model.save_model("test_model")

    model.setup_inference_model()
    model.evaluate_training_accuracy(100)

    test_x, test_y, max_target_length = model.generate_sample(test_file_path, 200)
    model.evaluate_model(test_x, test_y, 3, max_target_length, one_hot_encoder.model, "Test")




