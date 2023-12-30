import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

data = pd.read_csv("./Data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 결측치 -> 평균으로 대체
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
print(X)