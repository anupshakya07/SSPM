import time

import tensorflow as tf
import numpy as np
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import scipy as sc


class LearningModel:
    EMBEDDING_DIMENSION = 300
    TEST_SIZE = 0.1

    def __init__(self, latent_dim, kc_dimension):
        self.learning_rate = 0.001
        self.dropout_rate = 0.3
        self.loss_function = 'categorical_crossentropy'
        self.latent_dim = latent_dim
        self.kc_dimension = kc_dimension
        self.encoder_inputs = Input(shape=(None, self.EMBEDDING_DIMENSION), dtype=tf.float32)
        self.decoder_inputs = Input(shape=(None, self.kc_dimension), dtype=tf.float32)
        self.model = None
        self.encoder_states = []
        self.encoder_model = None
        self.decoder_model = None

    def create_model(self):
        encoder_lstm = LSTM(self.latent_dim, return_state=True)
        encoder_output, state_h, state_c = encoder_lstm(self.encoder_inputs)
        self.encoder_states = [state_h, state_c]

        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(self.decoder_inputs, initial_state=self.encoder_states)

        dropout_layer = Dropout(self.dropout_rate)
        decoder_outputs = dropout_layer(decoder_outputs)

        decoder_dense = Dense(self.kc_dimension, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=['acc', 'mse'])

    def set_embedding_dimension(self, dimension):
        self.EMBEDDING_DIMENSION = dimension

    def train_model(self, input_sequence, output_sequence, max_length_src, max_length_tar, num_epochs=30,
                    batch_size=100):
        x_train, x_test, y_train, y_test = train_test_split(input_sequence, output_sequence, test_size=0.1)

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

    def generate_batch(self, x, y, batch_size, input_dimension, output_dimension, max_length_src, max_length_tar):
        while True:
            for j in range(0, len(x), batch_size):
                encoder_input_data = np.zeros((batch_size, max_length_src, input_dimension), dtype='float32')
                decoder_input_data = np.zeros((batch_size, max_length_tar, output_dimension), dtype='float32')
                decoder_target_data = np.zeros((batch_size, max_length_tar, output_dimension), dtype='float32')

                masking_layer = tf.keras.layers.Masking(mask_value=np.zeros(self.kc_dimension))

                for i, (input_seq, target_seq) in enumerate(zip(x[j:j + batch_size], y[j:j + batch_size])):
                    for t, input_vec in enumerate(input_seq):
                        encoder_input_data[i, t] = input_vec
                    for t, output_vec in enumerate(target_seq):
                        if t < len(target_seq) - 1:
                            decoder_input_data[i, t] = output_vec
                        if t > 0:
                            decoder_target_data[i, t - 1] = output_vec
                masked_decoder_target_data = masking_layer(decoder_target_data)
                masked_decoder_input_data = masking_layer(decoder_input_data)
                yield [encoder_input_data, masked_decoder_input_data], masked_decoder_target_data

    def setup_inference_model(self):
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(self.decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states2 = [state_h2, state_c2]

        decoder_dense = Dense(self.kc_dimension, activation='softmax')
        decoder_outputs2 = decoder_dense(decoder_outputs2)

        self.decoder_model = Model([self.decoder_inputs] + decoder_state_inputs, [decoder_outputs2] + decoder_states2)

    def evaluate_model(self, x, y, max_length_src, max_length_tar, kcOneHotEncoder, eval_type="Training"):
        train_gen = self.generate_batch(x, y, batch_size=1, input_dimension=self.EMBEDDING_DIMENSION,
                                        output_dimension=self.kc_dimension, max_length_src=max_length_src,
                                        max_length_tar=max_length_tar)
        similarity_list = []
        num_correct = 0
        num_incorrect = 0
        start_time = time.time()
        for k in range(len(x)):
            (input_seq, actual_output), _ = next(train_gen)

            actual_output_words = []
            for ind in range(1, len(actual_output[0]) - 1):
                vector_np = actual_output[0][ind].numpy()
                is_zero_vector = (vector_np == np.zeros(self.kc_dimension)).all()
                is_one_vector = (vector_np == np.ones(self.kc_dimension)).all()

                if not is_zero_vector and not is_one_vector:
                    actual_output_words.append(kcOneHotEncoder.inverse_transform([vector_np])[0][0])
            #         print("Actual Words -> ", actual_output_words)
            res = 0
            decoded_sen = self.decode_sequence(input_seq, kcOneHotEncoder)
            if len(decoded_sen)>0:
                decoded_word_sentence = kcOneHotEncoder.inverse_transform(decoded_sen).reshape(-1)
            #         print("Predicted Sequence -> ", decoded_word_sentence)

                res = len(set(actual_output_words) & set(decoded_word_sentence)) / float(
                    len(set(actual_output_words) | set(decoded_word_sentence)))

            if res == 1:
                num_correct += 1
            else:
                num_incorrect += 1
            similarity_list.append(res)

        end_time = time.time()
        np_similarity_array = np.array(similarity_list)
        print("Latent Dimension = ", self.latent_dim)
        print("Total Correct = ", num_correct)
        print("Total Incorrect = ", num_incorrect)
        print("Similarity Mean on ", eval_type, " data = ", np_similarity_array.mean())
        print("Total Time taken for evaluation = ", end_time - start_time, " secs")

    def decode_sequence(self, input_seq, kcOneHotEncoder):
        start_token = np.ones(self.kc_dimension)
        end_token = np.ones(self.kc_dimension)

        state_values = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.kc_dimension))
        target_seq[0, 0] = start_token

        stop_condition = False
        decoded_sentence = []

        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + state_values)
            decoded_word = kcOneHotEncoder.inverse_transform([output_tokens[0][0]])

            if (cosine_similarity(output_tokens[0][0], end_token)) < 0.1 or len(decoded_sentence) > 100:
                stop_condition = True
            else:
                decoded_sentence.append(output_tokens[0][0])

            target_seq = np.zeros((1, 1, self.kc_dimension))
            corrected_one_hot = kcOneHotEncoder.transform(decoded_word)[0]
            target_seq[0][0] = corrected_one_hot
            state_values = [h, c]
        return decoded_sentence

    def save_model(self, model_name):
        self.model.save("saved_models/keras_models/"+model_name)

    def load_model(self, model_name):
        self.model = load_model("saved_models/keras_models/"+model_name)


def cosine_similarity(wv1, wv2):
    similarity = sc.spatial.distance.cosine(wv1, wv2)
    return similarity


if __name__ == "__main__":
    myModel = LearningModel(200, 933)
    myModel.create_model()
