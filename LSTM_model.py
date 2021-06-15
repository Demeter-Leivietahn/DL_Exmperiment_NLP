import pickle

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def load_embeddings(path):
    with open(path, 'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []

    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words


@logger.catch
def main():
    TRAIN = False
    df = pd.read_csv('data/train.txt')
    tokenizer = Tokenizer()
    df['review_text'] = df['review_text'].astype(str)
    tokenizer.fit_on_texts(df['review_text'])

    vocab_size = len(tokenizer.word_index)
    print('Vocabulary size:', vocab_size)
    sequences = tokenizer.texts_to_sequences(df['review_text'])
    padded_seq = pad_sequences(sequences, maxlen=1000, padding='post', truncating='post')
    X = padded_seq
    y = df['fit'].apply(lambda x: 0 if x == "small" else (2 if x == 'large' else (1 if x == 'fit' else x)))

    X_train, X_val, y_train, y_val = train_test_split(X, y.to_numpy(), test_size=0.2, stratify=y, random_state=2020)
    if TRAIN:
        GLOVE_EMBEDDING_PATH = 'data/glove.6B.300d.pkl'

        glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
        print('number unknown words (glove): ', len(unknown_words_glove))
        model1 = Sequential()

        embeddings = Embedding(vocab_size + 1, 300, weights=[glove_matrix], input_length=1000, trainable=False)

        model1.add(embeddings)
        model1.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
        model1.add(GlobalMaxPooling1D())
        model1.add(Dense(1, activation='softmax'))

        model1.summary()
        optimizer = Adam(0.001)

        model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model1.fit(X_train, y_train, batch_size=1024, epochs=1, validation_data=(X_val, y_val),
                   verbose=1)
        model1.save('lstm_model.tf', save_format='tf')
    else:
        model1 = load_model('lstm_model.h5')
    y_pred = model1.predict(X_val)
    classification_report(y_val, y_pred)


if __name__ == '__main__':
    main()
