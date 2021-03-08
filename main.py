from src import data_preprocessor
from src import one_hot_encoder
from src import model

if __name__ == "__main__":
    train_file_path = "datasets/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt"
    data_processor = data_preprocessor.DataPreprocessor(input_file_path=train_file_path)
    data_processor.analyze_dataset()

    one_hot_encoder = one_hot_encoder.OneHotEnc()
    one_hot_encoder.train(data_processor.unique_kcs)

    model = model.LearningModel(200, len(data_processor.unique_kcs))
    model.create_model()


