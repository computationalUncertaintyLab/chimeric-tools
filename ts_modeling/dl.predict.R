setwd("D:/workfiles/439")
source("memory.R")
source("auto.cov.R")


dl.predict=function(ar=0,ma=0,sigma2=1,tt=2){
  gamma=auto.cov(ar,ma,lag=tt-1)
  gamma0=gamma[1]
  rho=gamma[2:tt]/gamma0
  
  ## all calculation based on correlations
  Phi=array(0,c(tt-1,tt-1))
  V=rep(0,tt-1)
  Phi[1,1]=rho[1]
  V[1]=1-rho[1]^2
  for (m in 3:tt-1){
    Phi[m,m]= (rho[m] - sum(Phi[m-1,1:(m-1)]*rho[(m-1):1]))/V[m-1]
    Phi[m,1:(m-1)] = Phi[m-1,1:(m-1)] - Phi[m,m]*Phi[m-1,(m-1):1]
    V[m] = V[m-1]*(1-Phi[m,m]^2)
  }
  
  return(rbind(Phi,V*gamma0*sigma2))
}
