import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import src.tools as tools

stations = ['ParkersBrook','BrownsBrook','GrantPoole']
station_name = stations[0]

wd = f'/datadrive/stream_data/training/{station_name}'
bins = ['train', 'val', 'test']
data ={}
for b in bins:
    datatable_name = f'{wd}/{b}_table.csv'
    data[b] = pd.read_csv(datatable_name)
    #get data
    train_dataset = tools.create_classification_data_from_dataframe(data['train'])
    val_dataset = tools.create_classification_data_from_dataframe(data['val'])
    test_dataset = tools.create_classification_data_from_dataframe(data['test'])
    
