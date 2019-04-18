import numpy as np
import pandas as pd
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(description='Infer domain names from Tstat flow data')
parser.add_argument('--do_what', metavar='do_what', choices=['build_dataset', 'learn'])
parser.add_argument('--tstat_csv', metavar='tstat_csv', type=str, default='')
parser.add_argument('--tstat_npy', metavar='tstat_csv', type=str, default='')
parser.add_argument('--topn', metavar='topn', type=int, default=10)
parser.add_argument('--epochs', metavar='epochs', type=int, default=2)
globals().update(vars(parser.parse_args()))

indices = [ 'c_pkts_all:3', 's_pkts_all:17', 'c_bytes_uniq:7', 's_bytes_uniq:21', 'c_pkts_data:8', 's_pkts_data:22']

def main():
    if do_what == 'build_dataset':
        build_dataset(tstat_csv, topn, tstat_npy)
    else:
        learn(tstat_npy, len(indices), topn+1, epochs=epochs)

def choose_columns_for_x (df):
    x_obj = df.loc[:,indices].values
    x = x_obj.astype(np.float64)
    return x

def normalize_x(x):
    # check for nan values
    nisnanx = np.isnan(x).sum()
    if nisnanx > 0:
        for i in range(0, x.shape[0]):
            nn = np.isnan(x[i]).sum()
            if nn > 0:
                   print (i, x[i])
    # normalize [0, max] -> [0,1] ; use 1 if max is zero
    m = np.amax(x,0)
    for i in range(0,len(m)):
        if m[i] == 0 or np.isnan(m[i]):
            print (i, 'is', m[i])
            m[i] = 1
    # bring values closer to 1 by sqrt'ing them
    x = np.sqrt(x/m)
    return x

# Define class y according to the (y_class_count+1) you want 
def get_y(df, y_label_index, y_class_count):
    y_class_names = df[y_label_index].value_counts()[:y_class_count]
    y_class_names = y_class_names.index.to_list()
    y = df[y_label_index].values.copy()
    for i,s in enumerate(y):
        if s in y_class_names:
            y[i] = y_class_names.index(s) + 1
        else:
            # 'others' class
            y[i] = 0
    y = y.astype(np.int16)
    return y, y_class_names

# transform output to categorical for categorical loss
def to_categorical(y, n_cat):
    n_samples = y.shape[0]
    y_out = np.zeros((n_samples, n_cat))
    y_out[np.arange(n_samples), y] = 1
    return y_out

# set the same number of samples for all classess -- number of samples of the smallest class
def consider_class_imbalance(df, x, y, y_label_index, y_class_count):
    num_samples_per_class = df[y_label_index].value_counts()[y_class_count-1]
    v = []
    for l in np.unique(y):
        b = np.where(y == l)[0]
        if l == 0 and len(b) < num_samples_per_class:
            a = b
        else:
            a = np.random.choice(b, num_samples_per_class, replace=False)
        v.extend(a)
    mask = np.hstack([v])
    x = x[mask]
    y = y[mask]
    return x, y

# Split train-test 80% - 20% randomly
def split_train_test(x, y, train_p = 0.8):
    train_index = np.random.binomial(1, train_p, len(x))
    x_train = x[np.where(train_index == 1)]
    x_test = x[np.where(train_index == 0)]
    y_train = y[np.where(train_index == 1)]
    y_test = y[np.where(train_index == 0)]
    return x_train, y_train, x_test, y_test

def build_dataset(csvfilename, topn, npydatafile):
    df = pd.read_csv(csvfilename)
    x = choose_columns_for_x(df)
    x = normalize_x(x)
    y_label_index = 'c_tls_SNI:116'
    y, topn_names = get_y (df, y_label_index, topn)
    x, y = consider_class_imbalance(df, x, y, y_label_index, topn)
    x_train, y_train, x_test, y_test = split_train_test(x, y, 0.8)
    with open(npydatafile, "wb+") as fp:
        np.savez(fp, x_train, y_train, x_test, y_test, np.array(topn_names))     
        
def load_data (datafile):
    with open(datafile, "rb") as fp:
        nf = np.load(fp)
        test_train_data = []
        for k_i in range(0, len(nf.files)):
            test_train_data.append(nf['arr_' + str(k_i)])
    return test_train_data

def create_and_compile_model(num_inputs, num_outputs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2048, input_dim=num_inputs, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(num_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def learn(npydatafile, num_inputs, num_outputs, epochs=2):
    m = create_and_compile_model(num_inputs, num_outputs)
    x_train, y_train, x_test, y_test, _ = load_data(npydatafile)
    h = m.fit(x_train, to_categorical(y_train, num_outputs), epochs=epochs)
    m.evaluate(x_test, to_categorical(y_test, num_outputs), steps=4000)
    return h



# Main
if __name__ == '__main__':
    main()

