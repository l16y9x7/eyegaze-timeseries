"""
This file implements a hybrid of genetic algorithm with LSTM for the image manipulation-eye gaze timeseries data set.
It is designed to solve the classification problem about predicting the user's vote on if a picture is manipulated based
on a sequence of data on their eye gaze.
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

"""
Step 1: load data
"""
# load all data
data = pd.read_excel('Caldwell_ImageManipulation-EyeGaze_DataSetCombined.xlsx',
                        sheet_name='data')

data = data[["participant", "image", "image manipulated", "vote"]]

data_extended = pd.read_csv('Caldwell_Manip_Images_10-14_TimeSeries.csv')
# rename columns to make them align
data_extended = data_extended.rename(index=str, columns={'Participant_ID': 'participant', 'Image_ID': 'image'})
data = pd.merge(data_extended, data, how="left", on=["participant", "image"])    # join the dataframes
data.sort_values(by="Start Time")
print(data)
# input_data = data.drop(['image', 'image manipulated', 'vote'], axis=1)
# target_data = data['image manipulated']
#
# # normalise input data using z-score
# for column in range(input_data.shape[1]):
#     temp = input_data.iloc[:, column]
#     input_data.iloc[:, column].apply(lambda x: (x - temp.mean()) / temp.std())
