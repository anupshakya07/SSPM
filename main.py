from src import data_preprocessor

if __name__ == "__main__":
    train_file_path = "datasets/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt"
    data_processor = data_preprocessor.DataPreprocessor(input_file_path=train_file_path)
    data_processor.analyze_dataset()
