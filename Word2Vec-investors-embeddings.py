import random
import numpy as np
import pandas as pd
import time
import json
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import AveragePooling1D
from keras.layers.embeddings import Embedding
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


def main():
    print('loading and preparing data')
    file_name = 'investments.csv'
    num_rows = 168647
    data = load_data('investments.csv', num_rows) 
    
    investor_company_dict = create_investor_company_dict(data)
    selected_investors, df = select_investors(investor_company_dict, 5)
    selected_investor_company_dict = create_selected_new_dict(selected_investors, investor_company_dict)
    selected_companies = create_companies_set(selected_investors, selected_investor_company_dict)
    selected_investors_id_dict, selected_investors_id_dict_inverse = create_id_dict(selected_investor_company_dict)
    selected_companies_id_dict, selected_companies_id_dict_inverse = create_id_dict(selected_companies)
    
    # preprocess data
    labels = create_one_hot_labels(selected_investors_id_dict.values())
    encoded_inputs = encode_inputs(selected_investor_company_dict, selected_companies_id_dict)
    maxlen = df['number'].max()
    padded_inputs = pad_sequences(encoded_inputs, maxlen, padding='post')
    
    print('initializing neurual network')
    vocab_size = len(selected_companies)
    output_dim = len(selected_investors)
    _, model = train_model(name='model_1', num_epochs=1000, maxlen=maxlen,
    vocab_size=vocab_size, output_dim=output_dim, inputs=padded_inputs, labels=labels)
    print(model.summary())

    
    # print('start training')
    # experiments = []
    # for i in range(6):
    #     hist, model = train_model(num_epochs=10, name="model_{}".format(i), maxlen=maxlen,
    #         vocab_size=vocab_size, output_dim=output_dim, inputs=padded_inputs, labels=labels)
    #     experiments.append({'history': hist.history, 
    #                         'best_loss': min(hist.history['loss']), 
    #                         'best_acc': max(hist.history['acc'])})
    #     print(hist.history['acc'])

    #     with open('/Users/cicipan/projects/Predict-Success-of-Startups/results/experiments.json', 'w') as f:
    #         json.dump(experiments, f)




def load_data(file_name, num_rows):
    df = pd.read_csv(file_name)
#     if num_rows is None:
#         num_rows = len(df)    
    data = df.loc[:num_rows]
    return data



def create_investor_company_dict(dataframe):
    all_companies = set()
    all_investors = set()
    company_investor_dict = {}
    investor_company_dict = {}

    for i, row in dataframe.iterrows():
        company = row['company_name']
        investor = row['investor_name']

        if company not in all_companies:
            company_investor_dict[company] = set()
            company_investor_dict[company].add(investor)
        else:
            company_investor_dict[company].add(investor)

        if investor not in all_investors:
            investor_company_dict[investor] = set()
            investor_company_dict[investor].add(company)
        else:
            investor_company_dict[investor].add(company)

        all_companies.add(company)
        all_investors.add(investor)
    return investor_company_dict



def select_investors(investor_company_dict, min_portfolio_size):
# select the investors who invested more than 4 startups

    investor_portfolio_size = sorted(
        [(k, len(v)) for k, v in investor_company_dict.items() if len(v) >= min_portfolio_size], 
        reverse=True, key=lambda x: x[1])
    
    df_portfolio_size = pd.DataFrame(data=investor_portfolio_size, columns=['name', 'number'])
    
    selected_investors = set([x[0] for x in investor_portfolio_size])
    
    return selected_investors, df_portfolio_size
    



def create_selected_new_dict(selected_set, old_dict):
    new_dict = {}
    for i in selected_set:
        new_dict[i] = old_dict[i]
    return new_dict
        


def create_companies_set(selected_investors, selected_investor_company_dict):
    selected_companies = set()
    for i in selected_investors:
        companies = selected_investor_company_dict[i]
        for c in companies:
            selected_companies.add(c)
    return selected_companies



def create_id_dict(x):
    id_dict = {}
    id_dict_inverse = {}
    x = sorted(list(x))
    for i, k in enumerate(x):
        id_dict[k] = i
        id_dict_inverse[i] = k
    return id_dict, id_dict_inverse



def create_one_hot_labels(l):
    integer_encoded_labels = list(l)
    d = array(integer_encoded_labels)
    labels = to_categorical(d)
    return labels



def encode_inputs(i_c_new_dict, c_id_dict):
    vocab_size = len(c_id_dict)
    encoded_portfolios = []
    portfolios = list(i_c_new_dict.values())
    
    for i in portfolios:
        c_list = list(i)
        encoded_c_list = []
        for c in c_list:
            encoded_c_list.append(c_id_dict[c])
        
        encoded_portfolios.append(encoded_c_list)
    
    return encoded_portfolios 



def train_model(name, num_epochs, maxlen, vocab_size, output_dim, inputs, labels):

    companies_input = Input(shape=(maxlen,), dtype='int32')
    x = Embedding(vocab_size, 10, input_length=maxlen)(companies_input)
    x = AveragePooling1D(pool_size=(maxlen,))(x)
    x = Reshape((10,))(x)
    x = Dense(32)(x)
    output = Dense(output_dim, activation='softmax')(x)
    model = Model(companies_input, output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    model_json = model.to_json()
    with open('{}_gpu.json'.format(name), 'w') as json_file:
        json_file.write(model_json)

    checkpointer = ModelCheckpoint(
        filepath='{}_gpu_weights.hdf5'.format(name), 
        monitor='acc',
        verbose=1, save_best_only=True) 
    callbacks_list = [checkpointer]
    hist = model.fit(
        inputs, labels, epochs=num_epochs, verbose=0, callbacks=callbacks_list,
        batch_size=64)
    
    return hist, model


if __name__ == '__main__':
    main()

    
