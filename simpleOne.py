import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import seaborn as sns
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import scipy



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
    train['jobflag'].value_counts().plot(kind = 'bar')
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



def preProcessing(texts):
    stemmer = PorterStemmer()
    clean_texts = []
    for text in texts:
        # htmlタグを削除
        text = removeTag(text)
        # アルファベット以外をスペースに置き換え
        clean_punc = re.sub(r'[^a-zA-Z]', ' ', text)
        # 単語長が3文字以下のものは削除する
        clean_short_tokenized = [word for word in clean_punc.split() if len(word) > 3]
        # ステミング
        clean_normalize = [stemmer.stem(word) for word in clean_short_tokenized]
        # 単語同士をスペースでつなぎ, 文章に戻す
        clean_text = ' '.join(clean_normalize)
        clean_texts.append(clean_text)


    return clean_texts


def removeTag(x):
    p = re.compile(r"<[^>]*?>")
    return p.sub('', x)



def checkPreProcessing(combined ,combined_cleaned):
    print('#original\n', combined['description'][0])
    print("-----")
    print('#cleaned\n', combined_cleaned['description'][0])





def modeling(data, combined_cleaned):
    layers = keras.layers
    models = keras.models
    print("You have TensorFlow version", tf.__version__)


    train_size = int(len(data) * .8)
    print("Train size: %d" % train_size)
    print("Test size: %d" % (len(data) - train_size))
    # Train size: 1212
    # Test size: 304

    train_x = data[:train_size]
    test_x = data[train_size:]
    data_size = int(len(data))

    # tfidf_vector = TfidfVectorizer(
    #     input="array",
    #     norm="l2",
    #     max_features=None,
    #     sublinear_tf=True,
    #     stop_words="english",
    # )
    # tfid = tfidf_vector.fit_transform(combined_cleaned['description'])
    # delimit_num = data.shape[0]
    # train_main = tfid[:delimit_num]
    # test_y = tfid[delimit_num:]

    train_main = combined_cleaned[:data_size]
    test_y = combined_cleaned[data_size:]

    print(train_main.shape, test_y.shape)


    train_cat = train_x['jobflag']
    test_cat = test_x['jobflag']
    train_text = train_x['description']
    test_text = test_x['description']

    max_words = 1000
    tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(train_text)  # fit tokenizer to our training text data
    x_train = tokenize.texts_to_matrix(train_text)
    x_test = tokenize.texts_to_matrix(test_text)
    encoder = LabelEncoder()
    encoder.fit(train_cat)
    y_train = encoder.transform(train_cat)
    y_test = encoder.transform(test_cat)
    num_classes = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # print('x_train shape:', x_train.shape)
    # print('x_test shape:', x_test.shape)
    # print('y_train shape:', y_train.shape)
    # print('y_test shape:', y_test.shape)

    batch_size = 32
    epochs = 8
    drop_ratio = 0.1

    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(max_words,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(drop_ratio))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)

    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    return model, test_y, train_main




def submit(model, train_main, train, test_y,test):



    # model.fit(train_main, train['jobflag'])

    pred_sub = model.predict(test_y)
    sample_submit_df = pd.DataFrame([test['id'], pred_sub]).T
    sample_submit_df.to_csv('./sample2.csv', header=None, index=None)




def main():
    train, test = readData()
    checkData(train, test)
    # visualizeRawData(train, test)
    combined = train.append(test, ignore_index=True)
    combined_cleaned = combined.copy()
    combined_cleaned['description'] = preProcessing(combined['description'])
    train['description'] = preProcessing(train['description'])
    # combined_cleaned['description'] = combined_cleaned['description'].apply(lambda x: gensim.utils.simple_preprocess(x))
    # checkPreProcessing(combined, combined_cleaned)
    # bow = vectorlize(combined_cleaned)
    # train = labeling(train)
    model, test_y, train_main = modeling(train, combined_cleaned)
    submit(model, train_main, train, test_y, test)




if __name__ == '__main__':
    main()
