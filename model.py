import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv("hiring.csv")
dataset["experience"].fillna(0, inplace=True)
dataset["test_score"].fillna(dataset["test_score"].mean(), inplace=True)

