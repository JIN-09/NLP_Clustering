import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np





def readData():
    train = pd.read_csv("./train.csv")
    test = pd.read_csv("./test.csv")

    return train, test



def checkData(train, test):

    # print(train.head(30))
    # print(test.head(30))
    print(train.shape)  # 1516, 3
    print(test.shape)  # 1517, 2









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
