import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv("hiring.csv")

# Filling NaNs:
dataset["experience"].fillna('zero', inplace=True)
dataset["test_score"].fillna(dataset["test_score"].mean(), inplace=True)

# Selecting three columns of data:
scores = dataset.iloc[:, :3]
print(scores)

# Converting strings to integer values.
def convert_to_int(word):
	word_dict = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4,
	             'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9,
	             'ten':10, 'eleven':11, 'twelve':12}
	return word_dict[word]

scores['experience'] = scores['experience'].apply(lambda x : convert_to_int(x))

# Selecting salary from the last column.
salary = dataset.iloc[:, -1]

# Splitting training and test sets. Since the given dataset is
# small, we'll train our model with all the data at hand.
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

# Fitting the model with training data.
linreg.fit(scores, salary)

# Saving the model:
pickle.dump(linreg, open('model.pkl', 'wb'))

# Loading the model to compare the results.
model = pickle.load(open('model.pkl', 'rb'))
