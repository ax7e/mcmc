rm(list=ls())
setwd("/Users/haozhewen/Documents/GitHub/mcmc")
source('GameMCMC.R', chdir = TRUE)

library(ggplot2)
library(ggstance)
library(reshape2)
library(coda)
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
imp6 <- read.csv("data.csv")

## Analysis on important votes
X5 <- imp6[,-c(1)]
## Abstain as agree
X55 <- subset(X5, importantvote==1)
X56 <- X55[, -c(16:19,24:31)]
X66 <- na.omit(X56)

length(unique(X66$rcid))
## Note that the total number of important votes in the original data between 2001-2015 is 162
length(unique(X66$country))
unique(X66$year)
## index
year <- X66[,1]
country <- X66[,2]
## DVs
## the order is: US reward, US panish, comply, increase
## Two types of choices: abstain as agree or disagree; no CHN aid recipients as not receiving increase from China or missingness? namely, increaseA or increase

Y82 <- X66[, c(14, 15,  5, 17)]
X7 <- X66[, -c(1:22)]
## Abstain as non-compliance
colnc <- c(72, 68, 59,  64, 65, 69, 70,71, 73, 77, 78, 88)
colnr <-  c(48,44,  35, 40, 41, 45:47, 51, 53, 54, 1, 4, 5)
colnu <- c(24, 20, 11, 16, 17, 22, 23, 27, 29, 30, 80, 1,4, 5)
collistDA <- list(colnc,colnc,colnc,colnc,colnr,colnr,colnr,colnr,colnu,colnu,colnu,colnu)


# Print parameters
params <- list(Y=Y82, X=X7, year=year, country=country, covset=collistDA, burnin=1000, m=10000, h=0.01)
## cat("Parameters:\n")
## print(params)

Model2 <- GameMCMC12(Y=Y82, X=X7, year=year, country=country, covset=collistDA, burnin=1000, m=10000, h=0.01)
save(Model2, file="mcmcoutput/Model2.RData")
