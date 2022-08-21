import pandas as pd
import re
import numpy as np
from torch import optim
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch import nn
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#  Data Scientist (jobflag=1), Machine learning engieer(jobflag=2), Software engineer (jobflag=3), Consultant(jobflag=4)

def readData():
    train = pd.read_csv("./train.csv")
    test = pd.read_csv("./test.csv")

    return train, test


def checkData(train, test):

    # print(train.head(30))
    # print(test.head(30))
    print(train.shape) # 1516, 3
    print(test.shape) # 1517, 2




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




def modeling(combined_cleaned, train):



    tfidf_vector = TfidfVectorizer(
        input="array",
        norm="l2",
        max_features=None,
        sublinear_tf=True,
        stop_words="english",
    )
    tfid = tfidf_vector.fit_transform(combined_cleaned['description'])
    delimit_num = train.shape[0]
    train_x = tfid[:delimit_num, :]
    test_y = tfid[delimit_num:, :]

    print(train_x.shape, test_y.shape)

    x_train, x_test, y_train, y_test = train_test_split(train_x, train['jobflag'], test_size=.3, random_state=42)


    x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train)).float()
    x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()

    y_train = torch.tensor(y_train.values)
    y_test = torch.tensor(y_test.values)

    model = nn.Sequential(nn.Linear(x_train.shape[1], 64),
                         nn.ReLU(),
                         nn.Dropout(0.1),
                         nn.Linear(64, train['jobflag'].nunique()),
                         nn.LogSoftmax(dim=1))

    # Define the loss
    criterion = nn.NLLLoss()

    # Forward pass, get our logits
    logps = model(x_train)
    # Calculate the loss with the logits and the labels
    loss = criterion(logps, y_train)

    loss.backward()

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.002)


def evalGraph(history):


    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']);
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['train loss', 'test loss']);
    ax = plt.subplot(122)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['train accuracy', 'test accuracy']);
    plt.show()




def submit(pred, test):

    sample_submit_df = pd.DataFrame([test['id'], pred]).T
    sample_submit_df.to_csv('./sample.csv', header=None, index=None)




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


    modeling(combined_cleaned, train)
    # submit(pred, test)
    # evalGraph(history)





if __name__ == '__main__':
    main()
