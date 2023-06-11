# %%
import pandas as pd
import numpy as np
import pickle
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data
r_base = importr('base')
r_stats = importr('stats')

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


# %%
import numpy as np
from scipy.stats import norm,lognorm
import rpy2.robjects as robjects
from rpy2.robjects.vectors import FloatVector

def BTnormlog(x, mu, sigma=1):
    if x == 0:
        u = np.random.uniform()
        m = FloatVector([np.log(u)])+r_stats.pnorm(FloatVector([0]), FloatVector([mu]), sigma, log_p="True")
        xstar = r_stats.qnorm(m, FloatVector([mu]), sigma, log_p="True")
        xstar = np.array(xstar)[0]
    else:
        u = np.random.uniform()
        m = FloatVector([np.log(u)])+r_stats.pnorm(FloatVector([0]), FloatVector([-mu]), sigma, log_p="True")
        xstar = r_stats.qnorm(m, FloatVector([-mu]), sigma, log_p="True")
        xstar = -np.array(xstar)[0]
    return xstar
 
def BTnormlogv(x, mu, sigma=1):
    # if x is a Series, convert it to a numpy array
    if isinstance(x, pd.Series):
        x = x.values
    u = np.random.uniform(size=mu.shape)

    # Efficient way to do the calculation
    result = mu.copy()  # Create a copy of mu
    result[x == 1] *= -1  # Only multiply by -1 where x equals 1
    mu = FloatVector(result)
    m = FloatVector(np.log(u))+r_stats.pnorm(FloatVector(np.full(x.shape, 0)), mu, 1, log_p="True")
    xstar = r_stats.qnorm(m, mu, 1, log_p="True")
    xstar = np.array(xstar)[:x.shape[0]]
    xstar = xstar * (x==0) + (-xstar) * (x==1)
    return xstar


a0 = np.array([0,1,0],dtype=np.int64)
a2 = np.array([-400,-400,-400],dtype=np.float64)
sigma = np.array([1,1,1],dtype=np.float64)

BTnormlogv(a0, a2, sigma)

# %%
import numpy as np

def force_symmetric(x, uplo="U"):
    n = x.shape[0]
    if uplo == "U":
        mask = np.triu(np.ones((n, n), dtype=bool))
    elif uplo == "L":
        mask = np.tril(np.ones((n, n), dtype=bool))
    else:
        raise ValueError("Invalid value for uplo. Must be 'U' or 'L'.")
    return x * mask + x.T * (~mask)

# Example usage
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

result = force_symmetric(x, uplo="U")
print(result)


# %%
def GameMCMC12(Y, X, year, country, covset, m=100, burnin=10, h=0.001):
    print(f"m={m}")
    N = len(Y)
    
    ci1, ci2, ci3, ci4 = covset[:4]
    rc1, rc2, rcn1, rcn2 = covset[4:8]
    ur1, ur2, up1, up2 = covset[8:]

    k1, k2, k3, k4 = len(ci1), len(ci2), len(ci3), len(ci4)
    k5, k6, k7, k8 = len(rc1), len(rc2), len(rcn1), len(rcn2)
    k9, k10, k11, k12 = len(ur1), len(ur2), len(up1), len(up2)

    ## US reward given Recipient comply
    usr = Y.iloc[:,2] == 1
    USReward = Y[usr].iloc[:,0]
    Xu7R = X[usr].iloc[:,ur1]
    Xu9R = X[usr].iloc[:,ur2]
    year_usr = year[usr]
    country_usr = country[usr]
    Nusreward = len(USReward)

    ## US punish given Recipient not comply
    usp = Y.iloc[:,2]==0
    USPunish = Y[usp].iloc[:,1]
    Xu12P = X[usp].iloc[:,up1]
    Xu14P = X[usp].iloc[:,up2]
    year_usp = year[usp]
    country_usp = country[usp]
    Nuspunish = len(USPunish)

    # Recipient comply or not given China increase
    rci = Y.iloc[:,3]==1
    RComply = Y[rci].iloc[:,2]
    Xr7C = X[rci].iloc[:,rc1]
    Xr12C = X[rci].iloc[:,rc2]
    year_rci = year[rci]
    country_rci = country[rci]
    Nrcomply = len(RComply)

    # Recipient comply or not given China not increases
    rcni = Y.iloc[:,3]==0
    RNComply = Y[rcni].iloc[:,2]
    Xr9NC = X[rcni].iloc[:,rcn1]
    Xr14NC = X[rcni].iloc[:,rcn2]
    year_rcni = year[rcni]
    country_rcni = country[rcni]
    Nrncomply = len(RNComply)

    # China increase or not
    CIncrease = Y.iloc[:,3]
    Xc7 = X.iloc[:,ci1]
    Xc8 = X.iloc[:,ci2]
    Xc11 = X.iloc[:,ci3]
    Xc12 = X.iloc[:,ci4]
    ucountry_ci = np.unique(country)
    uyear_ci = np.unique(year)
    N_country = len(ucountry_ci)
    T_year = len(uyear_ci)

    betac7 = np.zeros(k1)
    betac8 = np.zeros(k2)
    betac11 = np.zeros(k3)
    betac12 = np.zeros(k4)

    betar7 = np.zeros(k5)
    betar9 = np.zeros(k6)
    betar12 = np.zeros(k7)
    betar14 = np.zeros(k8)

    betau7 = np.zeros(k9)
    betau9 = np.zeros(k10)    
    betau12 = np.zeros(k11)
    betau14 = np.zeros(k12)

    betar8 = 0
    betar10 = 0
    betac14 = 0 # intercept is put in the place of (-increase, not compliance, punish)



    def append_column(a, init, prepend=False):
        """
        Append a new column filled with a constant value to a numpy array.
        
        Parameters:
        a (numpy array): The original array.
        init (float, optional): The value to fill the new column with. Default is 1.

        Returns:
        numpy array: The original array with a new column appended.
        """
        col = np.full((a.shape[0], 1), init)
        out = np.hstack((a, col)) if not prepend else np.hstack((col, a))
        return out
    def init_p_from(x, init):
        return np.full((x.shape[0], 1), init)
    p1 = p2 = p3 = p4 = p5 = p6 = p7 = p8 = p9 = p10 = p11 = p12 = p13 = p14 = 0
    p7rc = p7rnc = p12rc = p12rnc = p8rc = p9rnc = p14rnc = p10rnc = 0
        # US reward given compliance
    Xuc = np.hstack((append_column(p1 * Xu7R, init = 1.0, prepend=True), p2 * Xu9R))
    betaucr = np.vstack((np.array([[0]]), betau7.reshape(-1, 1), betau9.reshape(-1,1)))
    newUusr = Xuc @ betaucr

    # US Punish given non-compliance
    Xunc = np.hstack((append_column(p1*Xu12P, init = 1.0, prepend=True), p2*Xu14P))
    betauncp = np.vstack((np.array([[0]]), betau12.reshape(-1,1), betau14.reshape(-1,1)))
    newUusp = Xunc @ betauncp

    # Recipient Comply given China increases
    # Xrc = np.hstack((p7*Xr7C, col, -p12 * Xr12C))
    Xrc = np.hstack((append_column(p7*Xr7C, init = 1.0, prepend=True), -p12*Xr12C))
    betarc = np.vstack((betar7.reshape(-1, 1), np.array([[betar8]]), betar12.reshape(-1, 1)))
    newUrc = Xrc @ betarc

    # Recipient Not Comply
    
    #Xrnc <- cbind(p9*Xr9NC, p10, -p14*Xr14NC)
    Xrnc = np.hstack((append_column(p9*Xr9NC, init = p10), -p14*Xr14NC))
    betarnc = np.vstack((betar9.reshape(-1, 1), np.array([[betar10]]), betar14.reshape(-1, 1)))
    newUrnc = Xrnc @ betarnc

    # China Increase
    # Xci <- cbind(p3*p7*Xc7, p3*p8*Xc8,
    #              p4*p11*Xc11, p4*p12*Xc12, -p6*p14)
    # col = np.full((Xc7.shape[0], 1), -p6*p14)
    Xci = append_column(np.hstack((p3*p7*Xc7, p3*p8*Xc8, p4*p11*Xc11, p4*p12*Xc12)), init=-p6*p14)
    # Íbetaci <- c(betac7, betac8, betac11, betac12, betac14)
    betaci = np.concatenate((betac7, betac8, betac11, betac12, np.array([betac14])), axis=0)
    # newUci <- as.matrix(Xci)%*% as.matrix(betaci)
    newUci = Xci @ betaci

    sr, sp, rc, rnc, ci = [len(x) for x in (betaucr, betauncp, betarc, betarnc, betaci)]

    Busr0, Busp0, Brc0, Brnc0, Bci0 = [np.eye(x) * h for x in (sr, sp, rc, rnc, ci)]

    # Create containers for MCMC draws
    betaUSReward = np.full((m, sr), np.nan)

    betaUSPunish = np.full((m, sp), np.nan)

    betaRComply = np.full((m, rc), np.nan)

    betaRNComply = np.full((m, rnc), np.nan)

    betaCIncrease = np.full((m, ci), np.nan)

    probabilityMedian = np.full((m, 5), np.nan)

    probabilitySD = np.full((m, 5), np.nan)

    alpha_precision = np.full((m, 1), np.nan)

    eta_precision = np.full((m, 1), np.nan)
    p1 = init_p_from(Xu7R, init = 0.0)
    p2 = init_p_from(Xu9R, init = 0.0)
    p7 = init_p_from(Xc7, init = 0.0)
    p11 = init_p_from(Xc11, init = 0.0)
    p12 = init_p_from(Xc12, init = 0.0)
    p9 = init_p_from(Xr9NC, init = 0.0)
    p10 = init_p_from(Xc11, init = 0.0)
    p14 = init_p_from(Xc11, init = 0.0)
    p8 = init_p_from(Xc8, init = 0.0)



    for g in range(1, m+burnin+1):
        if g % 10 == 0:
            print(f"g={g}")
        if g == m:
            break


        ## Analyze the conditional choice of China
        tt = np.c_[p3*p7*Xc7, p3*p8*Xc8, p4*p11*Xc11, p4*p12*Xc12]
        Xci = append_column(tt, init=-p6*p14)
        meanyci = Xci @ betaci

        newUci = meanyci.ravel()
        CutilityI = newUci
        # CIncrease_draw = np.array([BTnormlog(x=CIncrease.iloc[r], mu=CutilityI[r], sigma=1) for r in range(N)]) # replace with appropriate Python function
        CIncrease_draw = BTnormlogv(CIncrease, CutilityI)
        Bci1 = force_symmetric(np.linalg.pinv(Xci.T @ Xci + Bci0))
        mu_ci = Bci1 @ (Xci.T @ CIncrease_draw)
        betaci = np.random.multivariate_normal(mu_ci, Bci1, (1,)).T
        meanyci = Xci @ betaci
        newUci = meanyci.ravel()
        p1t = norm.cdf(newUci)
        p2t = 1-p1t
        p1sm = np.median(p1t)
        p1sd = np.std(p1t)

        p1 = p1t[usr]
        p2 = 1-p1
        p1r = p1t[usp]
        p2r = 1-p1r
        p1rc = p1t[rci]
        p2rc = 1-p1rc
        p1rnc = p1t[rcni]
        p2rnc = 1-p1rnc

        ################################################
        ## Analyze the conditional choice of Recipients
        ################################################
        ## Augment Data: Recipient's latent utility of complying  given China increase         

        Xrc = np.hstack((append_column(p7rc*Xr7C, init=p8rc), -p12rc*Xr12C))
        newUrc = Xrc @ betarc
        RutilityC = newUrc
        # RComply_draw = np.array([BTnormlog(x=RComply.iloc[r], mu=RutilityC[r], sigma=1) for r in range(Nrcomply)]) # replace with appropriate Python function
        RComply_draw = BTnormlogv(RComply, RutilityC)
        ## Update beta
        # Brc1 <- forceSymmetric(pseudoinverse(t(Xrc)%*%Xrc +Brc0)  )
        Brc1 = force_symmetric(np.linalg.pinv(Xrc.T @ Xrc + Brc0))
        # mu.urc <- Brc1%*%t(Xrc)%*%(Rcomply.draw)
        mu_urc = (Brc1 @ (Xrc.T @ RComply_draw)).reshape(-1)
        # betarc <-  mvrnorm(1, mu.urc, Brc1)
        betarc = np.random.multivariate_normal(mu_urc, Brc1, (1,)).T
        ## get the updated utility and probability of p3
        # newUrc <- as.numeric(Xrc%*%betarc)
        newUrc = Xrc @ betarc
        # p3s <- pnorm(newUrc)
        p3s = norm.cdf(newUrc)
        # p3sm <- mean(p3s)
        p3sm = np.mean(p3s)
        # p3sd <- sd(p3s)
        p3sd = np.std(p3s)
        ##*************************************
        ## Augment Data: Recipient's latent utility of not complying  given China not increase
        # Xrnc <- cbind(p9rnc*Xr9NC,p10rnc, -p14rnc*Xr14NC)
        Xrnc = np.hstack((append_column(p9rnc*Xr9NC, init=p10rnc), -p14rnc*Xr14NC))
        # newUrnc <- as.numeric(Xrnc%*%betarnc)
        newUrnc = Xrnc @ betarnc
        # RutilityNC <- newUrnc
        RutilityNC = newUrnc
        # RNcomply.draw <- sapply(c(1:Nrncomply),
        #                         function(r){BTnormlog(x=RNComply[r], mu= RutilityNC[r], sigma=1)})
        # RNComply_draw = np.array([BTnormlog(x=RNComply.iloc[r], mu=RutilityNC[r], sigma=1) for r in range(Nrncomply)]) # replace with appropriate Python function
        RNComply_draw = BTnormlogv(RNComply, RutilityNC)

        ## Update beta
        # Brnc1 <-  forceSymmetric(pseudoinverse(t(Xrnc)%*%Xrnc +Brnc0))  
        Brnc1 = force_symmetric(np.linalg.pinv(Xrnc.T @ Xrnc + Brnc0))
        # mu.urnc <- Brnc1%*%t(Xrnc)%*%(RNcomply.draw)
        mu_urnc = (Brnc1 @ (Xrnc.T @ RNComply_draw)).reshape(-1)
        # betarnc <- mvrnorm(1, mu.urnc, Brnc1)
        betarnc = np.random.multivariate_normal(mu_urnc, Brnc1, (1,)).T
        ## update the utility and probability of p5
        # newUrnc <- as.numeric(Xrnc%*%betarnc)
        newUrnc = Xrnc @ betarnc
        # p5s <- pnorm(newUrnc)
        p5s = norm.cdf(newUrnc)
        # p5sm <- mean(p5s)
        p5sm = np.mean(p5s)
        # p5sd <- sd(p5s) 
        p5sd = np.std(p5s)
        #############################################
        ##  analyze the conditional choice of US
        ############################################

        ## Augment Data: US latent utility of reawarding
        # Xuc <- data.frame(1, p1*Xu7R, p2*Xu9R) 
        Xuc = np.hstack((append_column(Xu7R*p1.reshape(-1,1), init = 1.0, prepend=True), Xu9R*p2.reshape(-1,1)))
        # Xuc <- as.matrix(Xuc)
        # newUusr <- as.numeric(Xuc%*%betaucr)
        newUusr = Xuc @ betaucr
        # USutilityR <- newUusr 
        USutilityR = newUusr
        # USreward.draw <- sapply(c(1:Nusreward),
        #                         function(r){BTnormlog(x=USReward[r], mu=USutilityR[r], sigma=1)})
        # USreward_draw = np.array([BTnormlog(x=USReward.iloc[r], mu=USutilityR[r], sigma=1) for r in range(Nusreward)]) # replace with appropriate Python function
        USreward_draw = BTnormlogv(USReward, USutilityR)

        ## Update beta
        # Busr1 <-  forceSymmetric(pseudoinverse(t(Xuc)%*%Xuc +Busr0))  
        Busr1 = force_symmetric(np.linalg.pinv(Xuc.T @ Xuc + Busr0))
        # mu.ucr <- Busr1%*%t(Xuc)%*%(USreward.draw)
        mu_ucr = (Busr1 @ (Xuc.T @ USreward_draw)).reshape(-1)
        # betaucr <- mvrnorm(1, mu.ucr, Busr1)
        betaucr = np.random.multivariate_normal(mu_ucr, Busr1, (1,)).T
        # newUusr <- as.numeric(Xuc%*%betaucr)
        newUusr = Xuc @ betaucr
        ## update the utility and probability of p7
        # p7s <- pnorm(newUusr)
        p7s = norm.cdf(newUusr)
        # p7sm <- median(p7s)
        p7sm = np.median(p7s)
        # p7sd <- sd(p7s)
        p7sd = np.std(p7s)

        ##**********************************************
        ## Augment Data: US latent utility of punishing
        # Xunc <- cbind(1, p1r*Xu12P, p2r*Xu14P)
        Xunc = np.hstack((append_column(Xu12P*p1r.reshape(-1,1), init=1.0, prepend=True), Xu14P*p2r.reshape(-1,1)))
        ## update the utility the probability of p12
        # newUusp <- as.numeric(Xunc%*%betauncp)
        newUusp = Xunc @ betauncp
        # USutilityP <- newUusp
        USutilityP = newUusp
        # USpunish.draw <- sapply(c(1:Nuspunish),
                                #function(r){BTnormlog(x=USPunish[r], mu=USutilityP[r], sigma=1)})
        # USpunish_draw = np.array([BTnormlog(x=USPunish.iloc[r], mu=USutilityP[r], sigma=1) for r in range(Nuspunish)]) # replace with appropriate Python function
        USpunish_draw = BTnormlogv(USPunish, USutilityP)

        ## Update beta
        # Busp1 <-  forceSymmetric(pseudoinverse(t(Xunc)%*%Xunc +Busp0))  
        Busp1 = force_symmetric(np.linalg.pinv(Xunc.T @ Xunc + Busp0))
        # mu.uncp <- Busp1%*%t(Xunc)%*%(USpunish.draw)
        mu_uncp = (Busp1 @ (Xunc.T @ USpunish_draw)).reshape(-1)
        # betauncp <- mvrnorm(1, mu.uncp, Busp1)
        betauncp = np.random.multivariate_normal(mu_uncp, Busp1, (1,)).T
        ## update the utility the probability of p12
        # newUusp <- as.numeric(Xunc%*%betauncp)
        newUusp = Xunc @ betauncp
        # p12s <- pnorm(newUusp)
        p12s = norm.cdf(newUusp)
        # p12sm <- median(p12s)
        p12sm = np.median(p12s)
        # p12sd <- sd(p12s)
        p12sd = np.std(p12s)

        ## Predict probabilities for the next round
        ## p7, p12
        
        # Xu7Rc <- X[,ur1]
        Xu7Rc = X.iloc[:,ur1]
        # Xu9Rc <- X[,ur2]
        Xu9Rc = X.iloc[:,ur2]

        # Xu12Pc <- X[,up1]
        Xu12Pc = X.iloc[:,up1]
        # Xu14Pc <- X[,up2]
        Xu14Pc = X.iloc[:,up2]

        # Xucc <-  data.frame(1, p1t*Xu7Rc, p2t*Xu9Rc)
        Xucc = np.hstack((append_column(Xu7Rc*p1t.reshape(-1,1), init=1.0, prepend=True), Xu9Rc*p2t.reshape(-1,1)))
        # Xucc <- as.matrix(Xucc)

        # Xuncc <- cbind(1, p1t*Xu12Pc, p2t*Xu14Pc)
        Xuncc = np.hstack((append_column(Xu12Pc*p1t.reshape(-1,1), init=1.0, prepend=True), Xu14Pc*p2t.reshape(-1,1)))

        # p7 <- pnorm(as.numeric(Xucc%*%betaucr))
        p7 = norm.cdf(Xucc @ betaucr)

        # p12 <- pnorm(as.numeric(Xuncc%*%betauncp))
        p12 = norm.cdf(Xuncc @ betauncp)
        '''
        p9 <- p7
        p8 <- 1-p7
        p10 <- p8
        p11 <- 1-p12
        p14 <- p12
        p13 <- 1-p14

        p7rc <- p7[rci]
        p7rnc <- p7[rcni]
        p12rc <- p12[rci]
        p12rnc <- p12[rcni]
        p8rc <- 1-p7rc
        p9rnc <- 1-p7rnc
        p14rnc <- p12rnc
        p10rnc <- 1-p7rnc
        '''
        p9 = p7
        p8 = 1-p7
        p10 = p8
        p11 = 1-p12
        p14 = p12
        p13 = 1-p14
        p7rc = p7[rci]
        p7rnc = p7[rcni]
        p12rc = p12[rci]
        p12rnc = p12[rcni]
        p8rc = 1-p7rc
        p9rnc = 1-p7rnc
        p14rnc = p12rnc
        p10rnc = 1-p7rnc
           
       
        '''
        ## predict p3, p5
        Xr7Cc <- X[,rc1]
        Xr12Cc <- X[,rc2]
        Xr9NCc <- X[,rcn1]
        Xr14NCc <- X[,rcn2]
        '''
        Xr7Cc = X.iloc[:,rc1]
        Xr12Cc = X.iloc[:,rc2]
        Xr9NCc = X.iloc[:,rcn1]
        Xr14NCc = X.iloc[:,rcn2]

        '''
        Xrcc <- cbind(p7*Xr7Cc, p8, -p12*Xr12Cc)
        Xrcc <- as.matrix(Xrcc)
        p3 <- pnorm(as.numeric(Xrcc%*%betarc))
        Xrncc <- cbind(p9*Xr9NCc,p10, -p14*Xr14NCc)
        Xrncc <- as.matrix(Xrncc)    
        p5 <- pnorm(as.numeric(Xrncc%*%betarnc))
        p4 <- 1-p3
        p6 <- 1-p5
        '''
        Xrcc = np.hstack((p7.reshape(-1,1)*Xr7Cc, p8.reshape(-1,1), -p12.reshape(-1,1)*Xr12Cc))
        p3 = norm.cdf(Xrcc @ betarc)
        Xrncc = np.hstack((p9.reshape(-1,1)*Xr9NCc, p10.reshape(-1,1), -p14.reshape(-1,1)*Xr14NCc))
        p5 = norm.cdf(Xrncc @ betarnc)
        p4 = 1-p3
        p6 = 1-p5
        # print all the ps
        # print(f"p1={p1}, p2={p2}, p3={p3}, p4={p4}, p5={p5}, p6={p6}, p7={p7}, p8={p8}, p9={p9}, p10={p10}, p11={p11}, p12={p12}, p13={p13}, p14={p14}")
        # print(f"p7rc={p7rc}, p7rnc={p7rnc}, p12rc={p12rc}, p12rnc={p12rnc}, p8rc={p8rc}, p9rnc={p9rnc}, p14rnc={p14rnc}, p10rnc={p10rnc}")
        if g > burnin:
            gg = g-burnin
            betaUSReward[gg,:] = betaucr.reshape(-1)
            betaUSPunish[gg,:] = betauncp.reshape(-1)
            betaRComply[gg,:] = betarc.reshape(-1)
            betaRNComply[gg,:] = betarnc.reshape(-1)
            betaCIncrease[gg,:] = betaci.reshape(-1)
            probmedia = np.hstack((p1sm, p3sm, p5sm, p7sm, p12sm))
            probabilityMedian[gg,:] = probmedia
            probsd = np.hstack((p1sd, p3sd, p5sd, p7sd, p12sd))


GameMCMC12(Y=Y82, X=X7, year=year, country=country, covset=collistDA, m=300, burnin=50, h=0.01) 


