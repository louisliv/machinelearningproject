import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import gensim

import scikitplot.plotters as skplt

import nltk

from xgboost import XGBClassifier

import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

from evaluate_classes import *

def main():
    df_train_txt = pd.read_csv('input/training_text', sep='\|\|', 
        header=None, skiprows=1, names=["ID","Text"])

    df_train_var = pd.read_csv('input/training_variants')

    df_test_txt = pd.read_csv('input/test_text', sep='\|\|', 
        header=None, skiprows=1, names=["ID","Text"])

    df_test_var = pd.read_csv('input/test_variants')

    df_train = pd.merge(df_train_var, df_train_txt, how='left', on='ID')


    df_test = pd.merge(df_test_var, df_test_txt, how='left', on='ID')

    df_train.describe(include='all')

    df_test.describe(include='all')

    df_train['Class'].value_counts().plot(kind="bar", rot=0)

    df_train, _ = train_test_split(df_train, test_size=0.7, 
        random_state=8, stratify=df_train['Class'])
    df_train.shape

    w2vec = get_word2vec(
        MySentences(
            df_train['Text'].values, 
            # df_test['Text'].values
        ),
        'w2vmodel'
    )

    mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2vec)
    mean_embedded = mean_embedding_vectorizer.fit_transform(df_train['Text'])

    evaluate_features(
        mean_embedded, 
        df_train['Class'].values.ravel(),
        XGBClassifier(
            max_depth=4,
            objective='multi:softprob',
            learning_rate=0.03333,
        )
    )

    num_words = 2000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df_train['Text'].values)
    X = tokenizer.texts_to_sequences(df_train['Text'].values)
    X = pad_sequences(X, maxlen=2000)

    embed_dim = 128
    lstm_out = 196

    # Model saving callback
    ckpt_callback = ModelCheckpoint('keras_model', 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='auto')

    model = Sequential()
    model.add(Embedding(num_words, embed_dim, input_length = X.shape[1]))
    model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(9,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', 
        metrics = ['categorical_crossentropy'])
    print(model.summary())

    Y = pd.get_dummies(df_train['Class']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
        test_size = 0.2, random_state = 42, stratify=Y)

    batch_size = 32
    model.fit(X_train, Y_train, epochs=8, batch_size=batch_size, 
        validation_split=0.2, callbacks=[ckpt_callback])

    model = load_model('keras_model')

    probas = model.predict(X_test)

    pred_indices = np.argmax(probas, axis=1)
    classes = np.array(range(1, 10))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], probas)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))
    skplt.plot_confusion_matrix(classes[np.argmax(Y_test, axis=1)], preds)

    # plt.savefig("mygraph.png")
    # plt.show()
    plot_model(model, to_file='model.png', show_shapes=True)

if __name__ == "__main__":
    main()