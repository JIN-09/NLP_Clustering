import gensim
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing import text, sequence
from keras_preprocessing.sequence import pad_sequences
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, SimpleRNN, Embedding, Dropout, SpatialDropout1D, Activation, Conv1D, GRU
from keras.layers import Conv1D, Bidirectional, GlobalMaxPool1D, MaxPooling1D, BatchNormalization, Add, Flatten
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
import keras.backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
import string
import re
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_words = 20000  # Max. workds to use per toxic comment
max_features = 200000  # Max. number of unique words in embeddinbg vector
max_len = 200  # Max. number of words per toxic comment to be use
embedding_dims = 128  # embedding vector output dimension
num_epochs = 15  # (before 5)number of epochs (number of times that the model is exposed to the training dataset)
val_split = 0.1
batch_size2 = 256


#  Data Scientist (jobflag=1), Machine learning engieer(jobflag=2), Software engineer (jobflag=3), Consultant(jobflag=4)

def readData():
    train = pd.read_csv("./train.csv")
    test = pd.read_csv("./test.csv")

    return train, test


def checkData(train, test):
    # print(train.head(30))
    # print(test.head(30))
    print(train.shape)  # 1516, 3
    print(test.shape)  # 1517, 2


def visualizeRawData(train, test):
    print(train['jobflag'].value_counts())
    train['jobflag'].value_counts().plot(kind='bar')
    plt.show()

    train_length = train['description'].str.len()
    test_length = test['description'].str.len()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.violinplot([train_length, test_length])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['train', 'test'])
    ax.set_ylabel('word count')
    plt.show()

    fig = plt.figure(figsize=(15, 4))
    for flag in [1, 2, 3, 4]:
        train_length_flag = train[train['jobflag'] == flag]['description'].str.len()
        ax = fig.add_subplot(1, 4, flag)
        ax.violinplot(train_length_flag)
        ax.set_xticks([1])
        ax.set_xticklabels([flag])
        ax.set_ylabel('word count')
    plt.tight_layout()
    plt.show()


def DataClean(texts):
    clean_texts = []
    for text in texts:
        # htmlタグを削除
        text = removeTag(text)
        # アルファベット以外をスペースに置き換え
        clean_punc = re.sub(r'[^a-zA-Z]', ' ', text)
        # 単語長が3文字以下のものは削除する
        # clean_short_tokenized = [word for word in clean_punc.split() if len(word) > 3]
        # # ステミング
        # clean_normalize = [stemmer.stem(word) for word in clean_short_tokenized]
        # 単語同士をスペースでつなぎ, 文章に戻す
        clean_text = ''.join(clean_punc)
        clean_texts.append(clean_text)

    return clean_texts


def removeTag(x):
    p = re.compile(r"<[^>]*?>")
    return p.sub('', x)


def checkDataClean(train, test):
    print('#train\n', train['description'][0])
    print("-----")
    print('#test\n', test['description'][0])


def preProcessing(train, test):
    X_train = train['description'].values
    X_test = test['description'].values
    y_train = train['jobflag'].values
    y_test = train['jobflag'].values
    print(y_test)

    # (before 32)The **batch size** is the number of training examples in one forward/backward pass.
    # In general, larger batch sizes result in faster progress in training, but don't always converge as quickly.
    # Smaller batch sizes train slower, but can converge faster. And the higher the batch size, the more memory space you’ll need.

    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(list(X_train))

    # Convert tokenized toxic commnent to sequnces
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # padding the sequences
    X_train = sequence.pad_sequences(X_train, max_len)
    X_test = sequence.pad_sequences(X_test, max_len)

    print('X_train shape:', X_train.shape)
    print('X_test shape: ', X_test.shape)

    X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, train_size=0.9, random_state=233)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=4)

    return X_tra, X_val, y_tra, y_val, X_train, X_test, y_train


def Word2VecModel(train):
    descripiton_lines = list()
    lines = train['description'].values.tolist()
    # nltk.download('punkt')
    # nltk.download('stopwords')

    for line in lines:
        tokens = word_tokenize(line)

        # convert to lower case
        tokens = [w.lower() for w in tokens]

        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # remove remaining tpkens gthat are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        descripiton_lines.append(words)

    # print(len(descripiton_lines))
    # 1516

    # train word2vec mode
    embedding_dims = 128  # embedding vector output dimension
    max_len = 200  # Max. number of words per toxic comment to be use
    word2VecModel = gensim.models.Word2Vec(sentences=descripiton_lines, vector_size=embedding_dims, window=5, workers=4,
                                           min_count=1)
    # vocab size
    words = list(word2VecModel.wv.index_to_key)
    print('Vocabulary size: %d' % len(words))  # 8827

    # save model
    # filename = './describ_embedding_word2vec.txt'
    # word2VecModel.wv.save_word2vec_format(filename, binary=False)

    word2Vec_embeddings_index = {}
    word2vec_file = open('./describ_embedding_word2vec.txt', encoding="utf-8")
    for line in word2vec_file:
        values = line.split()
        word = values[0]
        coefficient = np.asarray(values[1:])
        word2Vec_embeddings_index[word] = coefficient

    word2vec_file.close()

    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(descripiton_lines)
    sequences = tokenizer_obj.texts_to_sequences(descripiton_lines)
    word_index = tokenizer_obj.word_index
    print('Found %s uniquue tokens.' % len(word_index))

    Describ_pad = pad_sequences(sequences, maxlen=max_len)
    comments_tag = train['jobflag'].values
    print('Shape of toxic comments tensor', Describ_pad.shape)
    print('Shape of comment tensor', comments_tag.shape)

    num_words = len(word_index) + 200  # 8828
    word2Vec_embedding_matrix = np.zeros((num_words, embedding_dims))

    for word, i in word_index.items():
        if i > num_words:
            continue
        word2Vec_embedding_vector = word2Vec_embeddings_index.get(word)
        if word2Vec_embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            word2Vec_embedding_matrix[i] = word2Vec_embedding_vector

    print(num_words)
    print(word2Vec_embedding_matrix.shape[0])
    print(word2Vec_embedding_matrix.shape[1])

    return word2Vec_embedding_matrix


def CNN_Word2Vec_model(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train):
    CNN_Word2Vec_model = Sequential([
        Embedding(input_dim=word2Vec_embedding_matrix.shape[0], input_length=max_len,
                  output_dim=word2Vec_embedding_matrix.shape[1], weights=[word2Vec_embedding_matrix], trainable=False),
        SpatialDropout1D(0.5),
        # ... 100 filters with a kernel size of 4 so that each convolution will consider a window of 4 word embeddings
        Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'),
        # **batch normalization layer** normalizes the activations of the previous layer at each batch,
        # i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
        # It will be added after the activation function between a convolutional and a max-pooling layer.
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(6, activation='softmax'),
        Flatten()
    ])
    CNN_Word2Vec_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                               metrics=['accuracy', mean_pred, fmeasure, precision, recall])  # aucroc

    CNN_Word2Vec_model.summary()

    CNN_Word2Vec_model_fit = CNN_Word2Vec_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs,
                                                    validation_data=(X_val, y_val))

    CNN_Word2Vec_train_score = CNN_Word2Vec_model.evaluate(X_train, y_train, batch_size=batch_size2, verbose=1)
    print('Train Loss:', CNN_Word2Vec_train_score[0])
    print('Train Accuracy:', CNN_Word2Vec_train_score[1])
    print('Predicting....')
    y_pred = CNN_Word2Vec_model.predict(X_test, batch_size=batch_size2, verbose=1)


def RNN_Word2Vec_model(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train):
    RNN_Word2Vec_model = Sequential([
        Embedding(input_dim=word2Vec_embedding_matrix.shape[0], input_length=max_len,
                  output_dim=word2Vec_embedding_matrix.shape[1], weights=[word2Vec_embedding_matrix], trainable=False),
        SpatialDropout1D(0.5),
        # Fully-connected RNN where the output is to be fed back to input.
        SimpleRNN(25, return_sequences=True),
        # **batch normalization layer** normalizes the activations of the previous layer at each batch,
        # i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
        BatchNormalization(),
        Dropout(0.5),
        GlobalMaxPool1D(),
        Dense(50, activation='relu'),
        Dense(6, activation='softmax'),
        Flatten()

    ])

    RNN_Word2Vec_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                               metrics=['accuracy', mean_pred, fmeasure, precision])

    RNN_Word2Vec_model.summary()

    RNN_Word2Vec_model_fit = RNN_Word2Vec_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs,
                                                    validation_data=(X_val, y_val))

    RNN_Word2Vec_train_score = RNN_Word2Vec_model.evaluate(X_tra, y_tra, batch_size=batch_size2, verbose=1)
    print('Train Loss:', RNN_Word2Vec_train_score[0])
    print('Train Accuracy:', RNN_Word2Vec_train_score[1])

    return


def LSTMWidthWord2Vec(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train):
    LSTM_Word2Vec_model = Sequential([

        Embedding(input_dim=word2Vec_embedding_matrix.shape[0], input_length=max_len,
                  output_dim=word2Vec_embedding_matrix.shape[1], weights=[word2Vec_embedding_matrix], trainable=False),
        SpatialDropout1D(0.5),
        # Bidirectional layer will enable our model to predict a missing word in a sequence,
        # So, using this feature will enable the model to look at the context on both the left and the right.
        LSTM(25, return_sequences=True),
        # **batch normalization layer** normalizes the activations of the previous layer at each batch,
        # i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
        BatchNormalization(),
        Dropout(0.5),
        GlobalMaxPool1D(),
        Dense(50, activation='relu'),
        Dense(6, activation='softmax'),
        Flatten()
    ])

    LSTM_Word2Vec_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                                metrics=['accuracy', mean_pred, fmeasure, precision, recall])  # aucroc

    LSTM_Word2Vec_model.summary()
    LSTM_Word2Vec_model_fit = LSTM_Word2Vec_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs,
                                                      validation_data=(X_val, y_val))

    LSTM_Word2Vec_train_score = LSTM_Word2Vec_model.evaluate(X_train, y_train, batch_size=batch_size2, verbose=1)
    print('Train Loss:', LSTM_Word2Vec_train_score[0])
    print('Train Accuracy:', LSTM_Word2Vec_train_score[1])

    LSTM_Word2Vec_test_score = LSTM_Word2Vec_model.evaluate(X_test, X_test, batch_size=batch_size2, verbose=1)
    print('Test Loss:', LSTM_Word2Vec_test_score[0])
    print('Test Accuracy:', LSTM_Word2Vec_test_score[1])



def Bidirecitional_LSTM(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train):

    Bil_LSTM_Word2Vec_model = Sequential([
        Embedding(input_dim=word2Vec_embedding_matrix.shape[0], input_length=max_len,
                  output_dim=word2Vec_embedding_matrix.shape[1], weights=[word2Vec_embedding_matrix], trainable=False),
        SpatialDropout1D(0.5),
        # Bidirectional layer will enable our model to predict a missing word in a sequence,
        # So, using this feature will enable the model to look at the context on both the left and the right.
        Bidirectional(LSTM(25, return_sequences=True)),
        # **batch normalization layer** normalizes the activations of the previous layer at each batch,
        # i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
        BatchNormalization(),
        Dropout(0.5),
        GlobalMaxPool1D(),
        Dense(50, activation='relu'),
        Dense(6, activation='sigmoid'),
        Flatten()
    ])

    Bil_LSTM_Word2Vec_model.compile(loss='binary_crossentropy', optimizer=Adam(0.01),
                                    metrics=['accuracy', mean_pred, fmeasure, precision, recall]) #auroc,

    Bil_LSTM_Word2Vec_model.summary()

    Bil_LSTM_Word2Vec_model_fit = Bil_LSTM_Word2Vec_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs,
                                                              validation_data=(X_val, y_val))

    # train
    Bil_LSTM_Word2Vec_train_score = Bil_LSTM_Word2Vec_model.evaluate(X_train, y_train, batch_size=batch_size2,
                                                                     verbose=1)
    print('Train Loss:', Bil_LSTM_Word2Vec_train_score[0])
    print('Train Accuracy:', Bil_LSTM_Word2Vec_train_score[1])

    # test
    Bil_LSTM_Word2Vec_test_score = Bil_LSTM_Word2Vec_model.evaluate(X_test, X_test, batch_size=batch_size2, verbose=1)
    print('Test Loss:', Bil_LSTM_Word2Vec_test_score[0])
    print('Test Accuracy:', Bil_LSTM_Word2Vec_test_score[1])





def Gated_Recurrent(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train):

    sequence_input = Input(shape=(max_len,))
    model = Embedding(input_dim=word2Vec_embedding_matrix.shape[0], input_length=max_len,
                      output_dim=word2Vec_embedding_matrix.shape[1], weights=[word2Vec_embedding_matrix],
                      trainable=False)(sequence_input)
    model = SpatialDropout1D(0.2)(model)
    model = GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(model)
    model = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(model)
    avg_pool = GlobalAveragePooling2D()(model)
    max_pool = GlobalMaxPooling1D()(model)
    model = concatenate([avg_pool, max_pool])
    model = Flatten()(model)
    preds = Dense(6, activation="sigmoid")(model)

    GRU_Word2Vec_model = Model(sequence_input, preds)
    GRU_Word2Vec_model.compile(loss='binary_crossentropy', optimizer='adam',
                               metrics=['accuracy', mean_pred, fmeasure, precision, recall]) # auroc

    GRU_Word2Vec_model.summary()

    GRU_Word2Vec_model_fit = GRU_Word2Vec_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs,
                                                    validation_data=(X_val, y_val))






def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

        # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0.0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

    return fbeta_score


def submit(pred, test):
    sample_submit_df = pd.DataFrame([test['id'], pred]).T
    sample_submit_df.to_csv('./sample.csv', header=None, index=None)


@tf.function
def aucroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)


def main():
    train, test = readData()
    # checkData(train, test)
    # visualizeRawData(train, test)
    # combined = train.append(test, ignore_index=True)
    # combined_cleaned = combined.copy()
    # combined_cleaned['description'] = DataClean(combined['description'])
    train['description'] = DataClean(train['description'])
    test['description'] = DataClean(test['description'])
    # checkDataClean(train, test)
    X_tra, X_val, y_tra, y_val, X_train, X_test, y_train = preProcessing(train, test)
    word2Vec_embedding_matrix = Word2VecModel(train)
    CNN_Word2Vec_model(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train)
    # RNN_Word2Vec_model(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train)
    # LSTMWidthWord2Vec(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train)
    # Bidirecitional_LSTM(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train)
    # Gated_Recurrent(word2Vec_embedding_matrix, X_tra, X_val, y_tra, y_val, X_train, X_test, y_train)

    # submit(pred, test)


if __name__ == '__main__':
    main()
