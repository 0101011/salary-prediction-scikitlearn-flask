import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv("hiring.csv")
dataset["experience"].fillna(0, inplace=True)
dataset["test_score"].fillna(dataset["test_score"].mean(), inplace=True)

X = dataset.iloc[:, :3]

# Converting strings to integer values.
def convert_to_int(word):
	word_dict = ['zero':0, 'one':1, 'two':2, 'three':3, 'four':4,
	              'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9]
	return word_dict[word]
