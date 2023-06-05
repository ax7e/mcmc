import pandas as pd
import numpy as np
import pickle

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

import numpy as np
from scipy.stats import norm
import pandas as pd

def BTnormlog(x, mu, sigma=1):
    if x==0:
        u = np.random.uniform()
        xstar = norm.ppf(np.log(u) + norm.logpdf(0, mu, sigma))
    else:
        u = np.random.uniform()
        xstar = -norm.ppf(np.log(u) + norm.logpdf(0, -mu, sigma))
    return xstar

def GameMCMC12(Y, X, year, country, covset, m=10000, burnin=5000, h=0.001):
    N = len(Y)
    
    ci1, ci2, ci3, ci4 = covset[:4]
    rc1, rc2, rcn1, rcn2 = covset[4:8]
    ur1, ur2, up1, up2 = covset[8:]

    k1, k2, k3, k4 = len(ci1), len(ci2), len(ci3), len(ci4)
    k5, k6, k7, k8 = len(rc1), len(rc2), len(rcn1), len(rcn2)
    k9, k10, k11, k12 = len(ur1), len(ur2), len(up1), len(up2)
    
    usr = Y[Y[:,2] == 1]
    USReward = Y[usr, 0]
    Xu7R = X[usr[:,None],ur1]
    Xu9R = X[usr[:,None],ur2]
    year_usr = year[usr]
    country_usr = country[usr]
    Nusreward = len(usr)
   
    usp = Y[Y[:,2] == 0]
    USPunish = Y[usp, 1]
    Xu12P = X[usp[:,None],up1]
    Xu14P = X[usp[:,None],up2]
    year_usp = year[usp]
    country_usp = country[usp]
    Nuspunish = len(usp)

    rci = Y[Y[:,3] == 1]
    RComply = Y[rci,2]
    Xr7C = X[rci[:,None],rc1]
    Xr12C = X[rci[:,None],rc2]
    year_rci = year[rci]
    country_rci = country[rci]
    Nrcomply = len(rci)
  
    rcni = Y[Y[:,3] == 0]
    RNComply = Y[rcni,2]
    Xr9NC = X[rcni[:,None],rcn1]
    Xr14NC = X[rcni[:,None],rcn2]
    year_rcni = year[rcni]
    country_rcni = country[rcni]
    Nrncomply = len(rcni)
    
    CIncrease = Y[:,3]
    Xc7 = X[:,ci1]
    Xc8 = X[:,ci2]
    Xc11 = X[:,ci3]
    Xc12 = X[:,ci4]
    ucountry_ci = np.unique(country)
    uyear_ci = np.unique(year)
    N_country = len(ucountry_ci)
    T_year = len(uyear_ci)
    
    betac7 = np.zeros(k1)
    betac8 = np.zeros(k2)
    betac11 = np.zeros(k3)
    betac12 = np.zeros(k4)

