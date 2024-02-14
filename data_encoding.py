import os
import json
import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder 
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import datetime
import pickle

def temp():
    paths = []
    dirs = os.listdir('train_data')
    for dir in dirs:
        filenames = os.listdir(f'train_data/{dir}')
        paths += [f'train_data/{dir}/{filename}' for filename in filenames]
    paths = sorted(paths)
    return paths
data_filepaths = temp()

if not os.path.exists('./normalized_data'):
    os.mkdir('./normalized_data')
if not os.path.exists('./label_encoders'):
    os.mkdir('./label_encoders')


def encode_categoric_string_feature(feature_name):
    unique_ids = []
    for filepath in tqdm(data_filepaths, desc=feature_name.replace('_', ' ').capitalize()):
        df = pd.read_csv(filepath, usecols=[feature_name])
        df_unique_ids = df[feature_name].unique()
        del df
        unique_ids = np.unique(np.concatenate([unique_ids, df_unique_ids]))
        del df_unique_ids
    unique_ids_df = pd.DataFrame({
        feature_name: unique_ids
    })
    unique_ids_df.to_csv(f'./normalized_data/{feature_name}.csv', index=False)
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_ids)
    with open(f'./label_encoders/{feature_name}.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

def encode_categoric_string_feature_pandas(feature_name):
    unique_ids = None
    for filepath in tqdm(data_filepaths, desc=feature_name.replace('_', ' ').capitalize()):
        df = pd.read_csv(filepath, usecols=[feature_name])
        df_unique_ids = pd.Series(df[feature_name].unique())
        del df
        if unique_ids is None:
            unique_ids = df_unique_ids
        else:
            unique_ids = pd.Series(pd.concat([unique_ids, df_unique_ids]).unique())
        del df_unique_ids
    unique_ids_df = pd.DataFrame({
        feature_name: unique_ids
    })
    unique_ids_df.to_csv(f'./normalized_data/{feature_name}.csv', index=False)
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_ids)
    with open(f'./label_encoders/{feature_name}.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

features_string_categories = [
    'user_id_hash',
    'target_id_hash',
    'syndicator_id_hash',
    'campaign_id_hash',
    'target_item_taxonomy',
    'placement_id_hash',
    'publisher_id_hash',
    'source_id_hash',
    'source_item_type',
    'browser_platform',
]
for feature in features_string_categories:
    encode_categoric_string_feature(feature)

encode_categoric_string_feature_pandas('country_code')
encode_categoric_string_feature_pandas('region')

def normalize_file(encoders, filepath):
    main_dir = os.path.join('.', 'normalized_train_data')
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    
    df = pd.read_csv(filepath)

    for column, encoder in tqdm(encoders, desc=filepath):
        df[column] = encoder.transform(df[column].to_list())
    
    directory_path, filename = os.path.split(filepath)
    _, parent_dir = os.path.split(directory_path)
    parent_dir = os.path.join('.', 'normalized_train_data', parent_dir)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    df.to_csv(os.path.join(parent_dir, filename))

def load_encoders(features):
    column_encoders = []
    for feature in features:
        with open(f'./label_encoders/{feature}.pickle', 'rb') as handle:
            encoder = pickle.load(handle)
        column_encoders.append((feature, encoder))
    return column_encoders

features_string_categories = [
    'user_id_hash',
    'target_id_hash',
    'syndicator_id_hash',
    'campaign_id_hash',
    'target_item_taxonomy',
    'placement_id_hash',
    'publisher_id_hash',
    'source_id_hash',
    'source_item_type',
    'browser_platform',
    'country_code',
    'region',
]
column_encoders = load_encoders(features_string_categories)
for filepaths in data_filepaths:
    normalize_file(column_encoders, filepaths)