# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:11:14 2020

@author: fatih
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

training=pd.read_csv("energy_data_training.csv")
test=pd.read_csv("energy_data_test.csv")

x_train=training.drop(["Y1","Y2"],axis=1)
y_train=training.loc[:,["Y1","Y2"]]
x_test=test

cikti = LinearRegression().fit(x_train, y_train).predict(x_test)

print(cikti)

pd.DataFrame(cikti).to_csv("sonuc.csv",header=["Y1","Y2"],index=None)