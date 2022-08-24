import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split



#  Data Scientist (jobflag=1), Machine learning engieer(jobflag=2), Software engineer (jobflag=3), Consultant(jobflag=4)

def readData():
    train = pd.read_csv("./train.csv")
    test = pd.read_csv("./test.csv")

    return train, test


def checkData(train, test):

    print(train.head(30))
    print(test.head(30))
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
        clean_text = ''.join(clean_normalize)
        clean_texts.append(clean_text)


    return clean_texts


def removeTag(x):
    p = re.compile(r"<[^>]*?>")
    return p.sub('', x)



def checkPreProcessing(combined ,combined_cleaned):
    print('#original\n', combined['description'][0])
    print("-----")
    print('#cleaned\n', combined_cleaned['description'][0])





def labeling(train):

    le = LabelEncoder().fit(train['jobflag'])
    train['encoded_cat'] = le.transform(train['jobflag'])
    # print(train.head(10))


    return train



def sgd_pipeline():
    return Pipeline(
        [
            (
                "tfidf_vector_com",
                TfidfVectorizer(
                    input="array",
                    norm="l2",
                    max_features=None,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log",
                    penalty="l2",
                    class_weight='balanced',
                    tol=0.001,
                ),
            ),
        ]
    )

def svc_pipleline():
    return Pipeline(
        [
           (
                "clf",
                SVC(
                    C=10,
                    kernel="rbf",
                    gamma=0.1,
                    probability=True,
                    class_weight=None,
                ),
            ),
        ]
    )



def modeling(combined_cleaned, train):


    tfidf_vector =  TfidfVectorizer(
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



    svc_pipe = svc_pipleline()
    svc_pipe.fit(x_train, y_train)
    pred_test = svc_pipe.predict(x_test)
    pred_train = svc_pipe.predict(x_train)
    print_metrics(pred_test, y_test, pred_train, y_train)

    # sgd_pipe = sgd_pipeline()
    # sgd_pipe.fit(x_train, y_train)
    # pred_test = sgd_pipe.predict(x_test)
    # pred_train = sgd_pipe.predict(x_train)
    # print_metrics(pred_test, y_test, pred_train, y_train)




    return svc_pipe, train_x, train, test_y


def print_metrics(pred_test, y_test, pred_train, y_train):
    print("test accuracy", str(np.mean(pred_test == y_test)))
    print("train accuracy", str(np.mean(pred_train == y_train)))
    print("\n Metrics and Confusion for SVM \n")
    print(metrics.confusion_matrix(y_test, pred_test))
    print(metrics.classification_report(y_test, pred_test))



def submit(svc_pipe, train_x, train, test, test_y):

    svc_pipe.fit(train_x, train['jobflag'])
    pred_sub = svc_pipe.predict(test_y)
    sample_submit_df = pd.DataFrame([test['id'], pred_sub]).T
    sample_submit_df.to_csv('./sample.csv', header=None, index=None)



def main():
    train, test = readData()
    # checkData(train, test)
    # visualizeRawData(train, test)
    combined = train.append(test, ignore_index=True)
    combined_cleaned = combined.copy()
    combined_cleaned['description'] = preProcessing(combined['description'])
    train['description'] = preProcessing(train['description'])
    test['description'] = preProcessing(test['description'])
    # combined_cleaned['description'] = combined_cleaned['description'].apply(lambda x: gensim.utils.simple_preprocess(x))
    # checkPreProcessing(combined, combined_cleaned)

    # combined_cleaned = labeling(combined_cleaned)
    # train = labeling(train)
    svc_pipe, train_x, train, test_y= modeling(combined_cleaned, train)
    submit(svc_pipe, train_x, train, test, test_y)



if __name__ == '__main__':
    main()
