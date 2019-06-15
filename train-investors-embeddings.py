import random
import numpy as np
from numpy import array
from numpy import argmax
import pandas as pd
import time
import json
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint


def main():
    print('loading and preparing data')
    file_name = 'investments.csv'
    data = load_data('investments.csv') 
    
    PORTFOLIO_SIZE = 10
    MODEL_NAME = 'model_ps_{}'.format(PORTFOLIO_SIZE)

    investor_company_dict, investor_info_dict, _, company_info_dict = create_investor_company_dict(data)
    print(len(investor_company_dict))
    selected_investors, df = select_investors(investor_company_dict, PORTFOLIO_SIZE)
    print(len(selected_investors))
    selected_investor_company_dict = create_selected_new_dict(selected_investors, investor_company_dict)
    selected_companies = create_companies_set(selected_investors, selected_investor_company_dict)
    print(len(selected_companies))
    selected_investors_id_dict, selected_investors_id_dict_inverse = create_id_dict(selected_investor_company_dict)
    selected_companies_id_dict, selected_companies_id_dict_inverse = create_id_dict(selected_companies)

    # preprocess data
    labels = create_one_hot_labels(selected_investor_company_dict, selected_investors_id_dict)
    encoded_inputs = encode_inputs(selected_investor_company_dict, selected_companies_id_dict)
    maxlen = df['number'].max()
    padded_inputs = pad_sequences(encoded_inputs, maxlen, padding='post')

    print('initializing neurual network')
    vocab_size = len(selected_companies)
    output_dim = len(selected_investors)
    _, model = train_model(name=MODEL_NAME, num_epochs=1000, maxlen=maxlen,
    vocab_size=vocab_size, output_dim=output_dim, inputs=padded_inputs, labels=labels)
    print(model.summary())




def load_data(file_name):
    dt = pd.read_csv(file_name)
    dt = dt[['company_name','company_category_list','company_country_code','company_region',
         'investor_name','investor_permalink','investor_country_code','investor_region']]
    dt = dt.dropna()
    dt.info()
    return dt



def create_investor_company_dict(dataframe):
    all_companies = set()
    all_investors = set()
    company_investor_dict = {}
    investor_company_dict = {}
    company_info_dict = {} 
    # it should looks like ex: {company_name: {industry_cat: software, country_code: US, region: Bay Area}}
    investor_info_dict = {}
    # it should looks like ex: {investor_name: {country_code: US, region: Bay Area}}

    for i, row in dataframe.iterrows():
        company = row['company_name']
        industry_cat = row['company_category_list']
        comp_cc = row['company_country_code']
        comp_re = row['company_region']

        investor = row['investor_name']
        permalink = row['investor_permalink']
        investor_cc = row['investor_country_code']
        investor_re = row['investor_region']

        regex = re.match(r'/organization', permalink)
        if regex:
            if company not in all_companies:
                company_investor_dict[company] = set()
                company_investor_dict[company].add(investor)
                comp_info = {}
                comp_info['industry_cat'] = industry_cat
                comp_info['country_code'] = comp_cc
                comp_info['region'] = comp_re
                company_info_dict[company] = comp_info
            else:
                company_investor_dict[company].add(investor)

            if investor not in all_investors:
                investor_company_dict[investor] = set()
                investor_company_dict[investor].add(company)
                investor_info = {}
                investor_info['country_code'] = investor_cc
                investor_info['region'] = investor_re
                investor_info_dict[investor] = investor_info
            else:
                investor_company_dict[investor].add(company)

            all_companies.add(company)
            all_investors.add(investor)       
    return investor_company_dict, investor_info_dict, company_investor_dict, company_info_dict


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
#create a fixed id dictionary
    id_dict = {}
    id_dict_inverse = {}
    x = sorted(list(x))
    for i, k in enumerate(x):
        id_dict[k] = i
        id_dict_inverse[i] = k
    return id_dict, id_dict_inverse


def create_one_hot_labels(i_c_new_dict, i_id_dict):
    original = i_c_new_dict.keys()
    integer_encoded_labels = [i_id_dict[x] for x in original]
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




