###################################################################################################
##
## GameMCMC: R code for MCMC simulation of the three-player sequential game of aid for vote-buying
## Author: Xun Pang 
## Date: 05/02/2019
## Modified on 06/18/2019: add the sd of probabilities
## Description ---
## *************
## This code add an intercept of China's utility of betac14 to the version of GameMCMC2. R. Let China's utility of (not increase, not compliance, punishment) be different from other utilities in this path. The rationale is to let the probablities of p5 and p6 to have some impact on China's calculation of p1.
## The unit of observations is CHN-USA-Recipient-Resolution-Year, i.e., outcome of the three actors on a resolution at time t
## The dependent variable: Y =\{USreward, USpunish, Recompliance, CHNincrease\} with four outcome variables, each variable is a dichotomous one; for the votes in the same year, the other dependent variables except the compliance one are invariant
##******************
## Modified on May 14, 2019: add a time-specific and country-specific intercept in the China model
## Dataframes in different places. Note that there is not constant in any of the dataframe
## China's covariates
## Xc7       #  China increases, Rec. comply, and US award
## Xc8       #  China increases, Rec. comply, and US not award
## Xc11      #  China increases, Rec. not comply, and US not punish
## Xc12      #  China increases, Rec. not comply, and US punish
##
## Recipient's covariates
## Xr7       # China increases, Rec. comply, and US award
## Xr9       # China not increases, Rec. comply, and US award
## Xr12      # China increases, Rec. not comply, and US punish
## Xr14      # China not increase, Rec. not comply, and US punish
##
## US covariates
## Xu7       # China increases, Rec. comply, and US award
## Xu9       # China not increase, Rec. comply, and US award
## Xu12      # China increases, Rec. not comply, and US punish
## Xu14      # China not increases, Rec. not comply, and US punish
##
## Should create a list of covariates, assuming all those covariates have the same sample size
## CovariateList <- list(Xc7, Xc8, Xc11, Xc12, Xr7, Xr9, Xr12, Xr14, Xu7, Xu8, Xu12, Xu14)
## the excel data should look like GameDataDemo.xlsx
######################################################################################################

## load the required library

## load the required library
  library(corpcor)
 # library(mvtnorm)
  library(MASS)
#  library(msm)
library(Matrix)
#library(MCMCpack)

## functions
   BTnormlog <- function(x, mu, sigma=1){
    if(x==0) {
       u <- runif(1)
       xstar <- qnorm(log(u)+pnorm(0, mu, sigma, log=T), mu, sigma, log=T)
    } else{
       u <- runif(1)
       xstar <- -qnorm(log(u)+pnorm(0, -mu, sigma, log=T), -mu, sigma, log=T)
    }
    xstar
  }


  #######################################################################################
  ## MCMC Algorithm: Using data augmentation, but not rescale the
  ##          the augemented response variable (for small sample size)
  ## Description:
  ##          Y: a multi-variate variable, Y =\{USreward, USpunish, Recompliance, CHNincrease\}
##          X: a data frame containing all the covariates for different players
##          covset: a list specifying which column of x to be used as covariate of each utility of players
  ##          m: the numer of iteration after burnin
  ##          burnin: the numer of iterations discarded
  ##          nu.null: parameter to control how small  the variance is
  ##          Let's specify priors within the MCMC updating
  ##          h: the diagonal elelement of the inverse matrix of variance-covariance
  ##             of those beta priors
  ########################################################################################


## functions
   BTnormlog <- function(x, mu, sigma=1){
    if(x==0) {
       u <- runif(1)
       xstar <- qnorm(log(u)+pnorm(0, mu, sigma, log=T), mu, sigma, log=T)
    } else{
       u <- runif(1)
       xstar <- -qnorm(log(u)+pnorm(0, -mu, sigma, log=T), -mu, sigma, log=T)
    }
    xstar
  }

  BTnormlog2 <- function(x, mu, sigma=1){
    if(x==0) {
       u <- runif(1)
       u <- 0.1
       mu <- -400
       z <- log(u)+pnorm(0, mu, sigma, log=T)
       xstar <- qnorm(log(u)+pnorm(0, mu, sigma, log=T), mu, sigma, log=T)
    } else{
       u <- runif(1)
       u <- 0.1
       mu <- -400
       xstar <- -qnorm(log(u)+pnorm(0, -mu, sigma, log=T), -mu, sigma, log=T)
       z <- log(u)+pnorm(0, -mu, sigma, log=T)
    }
    xstar
  }



  GameMCMC12 <- function (Y, X, year, country, covset, m=10000, burnin=5000, h=0.001){
      ######################
      ## Arrange the data 
      ######################

      ## the sample size
      N <- nrow(Y)
      ## which covariates for which model?
      ci1<- covset[[1]]
      ci2<- covset[[2]]
      ci3<- covset[[3]]
      ci4<- covset[[4]]
      rc1 <- covset[[5]]
      rc2 <- covset[[6]]
      rcn1 <- covset[[7]]
      rcn2 <- covset[[8]]
      ur1 <- covset[[9]]
      ur2 <- covset[[10]]
      up1 <- covset[[11]]
      up2 <- covset[[12]]

      k1 <- length(ci1)
      k2 <- length(ci2)
      k3 <- length(ci3)
      k4 <- length(ci4)
      k5 <- length(rc1)
      k6 <- length(rc2)
      k7 <- length(rcn1)
      k8 <- length(rcn2)
      k9 <- length(ur1)
      k10 <- length(ur2)
      k11 <- length(up1)
      k12 <- length(up2)
      #### set the covariate matrix for each model
      ## US reward given Recipient comply
      usr <- which(Y[,3]==1)
      USReward <- Y[usr, 1]
      Xu7R <- X[usr,ur1]   
      Xu9R <- X[usr,ur2]
      year.usr <- year[usr]
      country.usr <- country[usr]
      Nusreward <- length(usr)
   
      ## US punish givan Recipient not comply
      usp <- which(Y[,3]==0)
      USPunish <- Y[usp, 2]
      Xu12P <- X[usp,up1]
      Xu14P <- X[usp,up2]
      year.usp <- year[usp]
      country.usp <- country[usp]
      Nuspunish <- length(usp)

      ## Recipient comply or not given China increase
      rci <- which(Y[,4]==1)
      RComply <- Y[rci,3]
      Xr7C <- X[rci,rc1]
      Xr12C <- X[rci,rc2]
      year.rci <- year[rci]
      country.rci <- country[rci]
      Nrcomply <- length(rci)
  
      ## Recipient comply or not given China not increases
      rcni <- which(Y[,4]==0)
      RNComply <- Y[rcni,3]
      Xr9NC <- X[rcni,rcn1]
      Xr14NC <- X[rcni,rcn2]
      year.rcni <- year[rcni]
      country.rcni <- country[rcni]
      Nrncomply <- length(rcni)
    
      ## China increase or not
      CIncrease <- Y[,4]
      Xc7 <- X[,ci1]
      Xc8 <- X[,ci2]
      Xc11 <- X[,ci3]
      Xc12 <- X[,ci4]
      ucountry.ci <- unique(country)
      uyear.ci <- unique(year)
      N.country <- length(ucountry.ci)
      T.year <- length(uyear.ci)
      ##################################################################
      ## Create parameters of beta's and p's and/or their initial values
      ##################################################################

      ## initial values 
      betac7 <- rep(0, k1)
      betac8 <- rep(0, k2)
      betac11 <- rep(0, k3)
      betac12 <- rep(0, k4)

      betar7 <- rep(0, k5)
      betar9 <- rep(0, k6)
      betar12 <- rep(0, k7)
      betar14 <- rep(0, k8)

      betau7 <- rep(0, k9)
      betau9 <- rep(0, k10)    
      betau12 <- rep(0, k11)
      betau14 <- rep(0, k12)

      betar8 <- 0
      betar10 <- 0
      betac14 <- 0 ## intercept is put in the place of (-increase, not compliance, punish)
      
      p1=p2=p3=p4=p5=p6=p7=p8=p7=p8=p9=p10=p11=p12=p13=p14 <-0
      p7rc =p7rnc =p12rc= p12rnc=p8rc =p9rnc= p14rnc =p10rnc <- 0
      ## US reward given compliance
      Xuc <- data.frame(1, p1*Xu7R, p2*Xu9R)  
      betaucr <- c(0, betau7, betau9)
      newUusr <- as.matrix(Xuc)%*% as.matrix(betaucr)
      ## US Punish given non-compliance
      Xunc <- cbind(1, p1*Xu12P, p2*Xu14P)
      betauncp <- c(0, betau12, betau14)
      newUusp <- as.matrix(Xunc)%*% as.matrix(betauncp)
      ## Recipient Comply given China increases
      Xrc <- cbind(p7*Xr7C, p8, -p12*Xr12C)
      betarc <- c(betar7, betar8, betar12)
      newUrc <- as.matrix(Xrc)%*% as.matrix(betarc)
      ## Recipient Not Comply
      Xrnc <- cbind(p9*Xr9NC, p10, -p14*Xr14NC)
      betarnc <- c(betar9, betar10, betar14)
      newUrnc <- as.matrix(Xrnc)%*% as.matrix(betarnc)
      ## China Increase
      Xci <- cbind(p3*p7*Xc7, p3*p8*Xc8,
                  p4*p11*Xc11, p4*p12*Xc12, -p6*p14)
      betaci <- c(betac7, betac8, betac11, betac12, betac14)
      newUci <- as.matrix(Xci)%*% as.matrix(betaci)

      ## create variance-covariance matrices of the beta prior
      ## assume beta priors are multivariate normal with zero mean
      ## and diagonal variance-covariance matrix
      sr <- length(betaucr)
      Busr0 <- diag(h, sr)

      sp <- length(betauncp)
      Busp0 <- diag(h, sp)

      rc <- length(betarc)
      Brc0 <- diag(h, rc)

      rnc <- length(betarnc)
      Brnc0 <- diag(h, rnc)

      ci <- length(betaci)
      Bci0 <- diag(h, ci)
      

      ## Create containers for MCMC draws
      betaUSReward <- matrix (NA, ncol=sr, nrow=m)
      betaUSPunish <- matrix(NA, ncol=sp, nrow=m)
      betaRComply <- matrix(NA, ncol=rc, nrow=m)
      betaRNComply <- matrix(NA, ncol=rnc, nrow=m)
      betaCIncrease <- matrix(NA, ncol=ci, nrow=m)
      probabilityMedian <- matrix(NA, ncol=5, nrow=m)
      probabilitySD <- matrix(NA, ncol=5, nrow=m)
      alpha.precision <- matrix(NA, ncol=1, nrow=m)
      eta.precision <- matrix(NA, ncol=1, nrow=m)
     #########################
     ## loop begins
     #########################
     
     #Tracking the process of MCMC simulation
     for(g in 1:(m+burnin)) { 
       if (g%%10 ==0){
         cat("g=",g,"\n")
       }
       if (g==m){
         break
       }

       ################################################
       ## Analyze the conditional choice of China
       ################################################
       ## Augment Data: China's latent utility of increasing aid to a recipient
       Xci <- cbind(p3*p7*Xc7, p3*p8*Xc8, p4*p11*Xc11, p4*p12*Xc12,  -p6*p14)
       Xci <- as.matrix(Xci)
       meanyci <- Xci%*%betaci
       newUci <- as.numeric(meanyci)
       CutilityI <- newUci
       CIncrease.draw <- sapply(c(1:N),
                               function(r){BTnormlog(x=CIncrease[r], mu=CutilityI[r], sigma=1)})
       ## Update beta
       Bci1 <- forceSymmetric(pseudoinverse(t(Xci)%*%Xci +Bci0))      
       mu.ci <- Bci1%*%t(Xci)%*%((CIncrease.draw))
       betaci <- mvrnorm(1, mu.ci, Bci1)
       meanyci <- Xci%*%betaci
       ## update the utility and probablity of p1
       newUci <- as.numeric(meanyci)
       p1t <- pnorm(newUci)
       p2t <- 1-p1t
       p1sm <- median(p1t)
       p1sd <- sd(p1t)
       
       p1 <- p1t[usr]
       p2 <- 1-p1
       
       p1r <- p1t[usp]
       p2r <- 1-p1r

       p1rc <- p1t[rci]
       p2rc <- 1-p1rc

       p1rnc <- p1t[rcni]
       p2rnc <- 1-p1rnc

       ################################################
       ## Analyze the conditional choice of Recipients
       ################################################
       ## Augment Data: Recipient's latent utility of complying  given China increase         
       Xrc <- cbind(p7rc*Xr7C, p8rc, -p12rc*Xr12C)
       Xrc <- as.matrix(Xrc)
       newUrc <- as.numeric(Xrc%*%betarc)
       RutilityC <- newUrc
       Rcomply.draw <- sapply(c(1:Nrcomply),
                               function(r){BTnormlog(x=RComply[r], mu=RutilityC[r], sigma=1)})

       ## Update beta
       Brc1 <- forceSymmetric(pseudoinverse(t(Xrc)%*%Xrc +Brc0)  )
       mu.urc <- Brc1%*%t(Xrc)%*%(Rcomply.draw)
       betarc <-  mvrnorm(1, mu.urc, Brc1)
       ## get the updated utility and probability of p3
       newUrc <- as.numeric(Xrc%*%betarc)
       p3s <- pnorm(newUrc)
       p3sm <- mean(p3s)
       p3sd <- sd(p3s)
       ##*************************************
       ## Augment Data: Recipient's latent utility of not complying  given China not increase
       Xrnc <- cbind(p9rnc*Xr9NC,p10rnc, -p14rnc*Xr14NC)
       Xrnc <- as.matrix(Xrnc)
       newUrnc <- as.numeric(Xrnc%*%betarnc)
       RutilityNC <- newUrnc
       RNcomply.draw <- sapply(c(1:Nrncomply),
                               function(r){BTnormlog(x=RNComply[r], mu= RutilityNC[r], sigma=1)})

       ## Update beta
       Brnc1 <-  forceSymmetric(pseudoinverse(t(Xrnc)%*%Xrnc +Brnc0))  
       mu.urnc <- Brnc1%*%t(Xrnc)%*%(RNcomply.draw)
       betarnc <- mvrnorm(1, mu.urnc, Brnc1)
       ## update the utility and probability of p5
       newUrnc <- as.numeric(Xrnc%*%betarnc)
       p5s <- pnorm(newUrnc)
       p5sm <- mean(p5s)
       p5sd <- sd(p5s) 

       #############################################
       ##  analyze the conditional choice of US
       ############################################

       ## Augment Data: US latent utility of reawarding
       Xuc <- data.frame(1, p1*Xu7R, p2*Xu9R) 
       Xuc <- as.matrix(Xuc)
       newUusr <- as.numeric(Xuc%*%betaucr)
       USutilityR <- newUusr 
       USreward.draw <- sapply(c(1:Nusreward),
                               function(r){BTnormlog(x=USReward[r], mu=USutilityR[r], sigma=1)})

       ## Update beta
       Busr1 <-  forceSymmetric(pseudoinverse(t(Xuc)%*%Xuc +Busr0))  
       mu.ucr <- Busr1%*%t(Xuc)%*%(USreward.draw)
       betaucr <- mvrnorm(1, mu.ucr, Busr1)
       newUusr <- as.numeric(Xuc%*%betaucr)
       ## update the utility and probability of p7
       p7s <- pnorm(newUusr)
       p7sm <- median(p7s)
       p7sd <- sd(p7s)
       ##**********************************************
       ## Augment Data: US latent utility of punishing
       Xunc <- cbind(1, p1r*Xu12P, p2r*Xu14P)
       Xunc <- as.matrix(Xunc)
       ## update the utility the probability of p12
       newUusp <- as.numeric(Xunc%*%betauncp)
       USutilityP <- newUusp
       USpunish.draw <- sapply(c(1:Nuspunish),
                               function(r){BTnormlog(x=USPunish[r], mu=USutilityP[r], sigma=1)})

       ## Update beta
       Busp1 <-  forceSymmetric(pseudoinverse(t(Xunc)%*%Xunc +Busp0))  
       mu.uncp <- Busp1%*%t(Xunc)%*%(USpunish.draw)
       betauncp <- mvrnorm(1, mu.uncp, Busp1)
       ## update the utility the probability of p12
       newUusp <- as.numeric(Xunc%*%betauncp)
       p12s <- pnorm(newUusp)
       p12sm <- median(p12s)
       p12sd <- sd(p12s)

       ## Predict probabilities for the next round
       ## p7, p12
       
       Xu7Rc <- X[,ur1]
       Xu9Rc <- X[,ur2]
       Xu12Pc <- X[,up1]
       Xu14Pc <- X[,up2]
       Xucc <-  data.frame(1, p1t*Xu7Rc, p2t*Xu9Rc)
       Xucc <- as.matrix(Xucc)
       Xuncc <- cbind(1, p1t*Xu12Pc, p2t*Xu14Pc)
       Xuncc <- as.matrix(Xuncc)
       p7 <- pnorm(as.numeric(Xucc%*%betaucr))
       p12 <- pnorm(as.numeric(Xuncc%*%betauncp))
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
           
       
       ## predict p3, p5
       Xr7Cc <- X[,rc1]
       Xr12Cc <- X[,rc2]
       Xr9NCc <- X[,rcn1]
       Xr14NCc <- X[,rcn2]
       Xrcc <- cbind(p7*Xr7Cc, p8, -p12*Xr12Cc)
       Xrcc <- as.matrix(Xrcc)
       p3 <- pnorm(as.numeric(Xrcc%*%betarc))
       Xrncc <- cbind(p9*Xr9NCc,p10, -p14*Xr14NCc)
       Xrncc <- as.matrix(Xrncc)    
       p5 <- pnorm(as.numeric(Xrncc%*%betarnc))
       p4 <- 1-p3
       p6 <- 1-p5
       
       #######################
       ## Save draws
       #######################
       if (g > burnin){
           gg <- g-burnin
           betaUSReward[gg,] <- as.vector(betaucr)
           betaUSPunish[gg,] <- as.vector(betauncp)
           betaRComply[gg,] <-  as.vector(betarc)
           betaRNComply[gg,] <-  as.vector(betarnc)
           betaCIncrease[gg,] <- as.vector(betaci)
           probmedia <- c(p1sm, p3sm, p5sm, p7sm, p12sm)
           probabilityMedian[gg,] <- probmedia
           probsd <- c(p1sd, p3sd, p5sd, p7sd, p12sd)
           probabilitySD[gg,] <- probsd
       }
       }
    #########################
    ## Return the MCMC output
    #########################
   return(list(betaUSReward= betaUSReward,
               betaUSPunish=betaUSPunish,betaRComply=betaRComply,
               betaRNComply=betaRNComply, betaCIncrease=betaCIncrease,
               probabilityMedian=probabilityMedian, probabilitySD=probabilitySD))
   }                                                      


##### I will write another function to calculate the posterior probabilities and their percentage change compare to the probabilities when holding all things at the median. The nice thing is that I can get all the uncertainty including the probabilities and the percentage change!!!
