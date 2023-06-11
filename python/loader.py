import pandas as pd
import numpy as np
import pandas as pdmport pickle
from scipy.stats import norm

# Read the data
imp6 = pd.read_csv("../data.csv")

# Analysis on important votes
X5 = imp6.drop(imp6.columns[0], axis=1)

# Abstain as agree
X55 = X5[X5['importantvote'] == 1]
X56 = X55.drop(X55.columns[15:19], axis=1)
X57 = X56.drop(X55.columns[23:31], axis=1)
X66 = X57.dropna()

print(len(np.unique(X66['rcid'])))
## Note that the total number of important votes in the original data between 2001-2015 is 162
print(len(np.unique(X66['country'])))
print(np.unique(X66['year']))

# Index
year = X66.iloc[:,0]
country = X66.iloc[:,1]

# DVs
# the order is: US reward, US panish, comply, increase
## Two types of choices: abstain as agree or disagree; no CHN aid recipients as not receiving increase from China or missingness? namely, increaseA or increase
Y82 = X66.iloc[:, [13, 14, 4, 16]]
X7 = X66.drop(X66.columns[range(0,22)], axis=1)

# Abstain as non-compliance
colnc = [71, 67, 58, 63, 64, 68, 69,70, 72, 76, 77, 87]
colnr = [47,43, 34, 39, 40, 44,45,46, 50, 52, 53, 0, 3, 4]
colnu = [23, 19, 10, 15, 16, 21, 22, 26, 28, 29, 79, 0, 3, 4]
collistDA = [colnc, colnc, colnc, colnc, colnr, colnr, colnr, colnr, colnu, colnu, colnu, colnu]

