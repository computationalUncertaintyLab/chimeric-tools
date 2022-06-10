setwd("D:/workfiles/439")
source("memory.R")
source("auto.cov.R")
source("dl.predict.R")


I0.predict=function(x,ar=0,ma=0,mu=0,origin=1,h=1,sigma2=1){
  y=x-mu
  tt=origin+h
  V=dl.predict(ar=ar,ma=ma,tt=tt)
  
  Phi=diag(h)
  for (j in 2:h){
    Phi[j,1:(j-1)]=-V[origin-2+j,(j-1):1]
  }
  ErrCov=solve(Phi)%*%diag(V[tt,(tt-h):(tt-1)])%*%t(solve(Phi))
  
  y.pred=rep(0,tt)
  y.pred[1:origin]=y[1:origin]
  y.se=sqrt(diag(ErrCov))
  for (j in 1:h){
    y.pred[origin+j] = sum(V[origin-1+j,1:(origin-1+j)]*y.pred[(origin-1+j):1])
  }
  
  sd=sqrt(sigma2)
  return(rbind(y.pred[origin+1:h]+mu,y.se*sd))
}



I1.predict=function(x,ar=0,ma=0,mu=0,origin=2,h=1,sigma2=1){
  y=diff(x)-mu
  tt=origin+h-1
  V=dl.predict(ar=ar,ma=ma,tt=tt)
  
  Phi=diag(h)
  for (j in 2:h){
    Phi[j,1:(j-1)]=-V[origin-2+j,(j-1):1]
  }
  ErrCov=solve(Phi)%*%diag(V[tt,(tt-h):(tt-1)])%*%t(solve(Phi))
  
  y.pred=rep(0,tt)
  y.pred[1:(origin-1)]=y[1:(origin-1)]
  x.pred=rep(0,tt+1)
  x.pred[1:origin] = x[1:origin]
  y.se=sqrt(diag(ErrCov))
  x.se=rep(0,h)
  for (j in 1:h){
    y.pred[origin-1+j] = sum(V[origin-2+j,1:(origin-2+j)]*y.pred[(origin-2+j):1])
    x.pred[origin+j] = x[origin] + sum(y.pred[origin:(origin-1+j)]) + j*mu
    temp=rep(0,h)
    temp[1:j]=1
    x.se[j]=sqrt(t(temp)%*%ErrCov%*%temp)
  }
  
  sd=sqrt(sigma2)
  return(rbind(x.pred[origin+1:h],x.se*sd,y.pred[origin-1+1:h]+mu,y.se*sd))
}


###################### test the functions
## set.seed(1)
## x=arima.sim(list(ar=.5,ma=.2),n=500)
## x=cumsum(x)
## y=diff(x)
## fit.y=arima(y,order=c(1,0,1))
## pred5=I1.predict(x,ar=fit.y$coef[1],ma=fit.y$coef[2],mu=fit.y$coef[3],origin=500,h=5,sigma2=fit.y$sigma2)
## predict(fit.y,5)
## I0.predict(y,ar=fit.y$coef[1],ma=fit.y$coef[2],mu=fit.y$coef[3],origin=499,h=5,sigma2=fit.y$sigma2)
## pred5
## pred5[1,2:5]-pred5[1,1:4]
## pred200=I1.predict(x,ar=fit.y$coef[1],ma=fit.y$coef[2],mu=fit.y$coef[3],origin=500,h=200,sigma2=fit.y$sigma2)
## gamma=auto.cov(ar=fit.y$coef[1],ma=fit.y$coef[2],lag=200-1,sigma2=fit.y$sigma2)
## sum(gamma*200:1)*2-200*gamma[1]
## pred200[2,200]^2

## > set.seed(1)
## > x=arima.sim(list(ar=.5,ma=.2),n=500)
## > x=cumsum(x)
## > y=diff(x)
## > fit.y=arima(y,order=c(1,0,1))
## > pred5=I1.predict(x,ar=fit.y$coef[1],ma=fit.y$coef[2],mu=fit.y$coef[3],origin=500,h=5,sigma2=fit.y$sigma2)
## > predict(fit.y,5)
## $pred
## Time Series:
## Start = 500 
## End = 504 
## Frequency = 1 
## [1] -1.8905185 -0.8468187 -0.3700636 -0.1522851 -0.0528054

## $se
## Time Series:
## Start = 500 
## End = 504 
## Frequency = 1 
## [1] 1.019424 1.227206 1.266270 1.274270 1.275933

## > pred5
##           [,1]       [,2]       [,3]       [,4]       [,5]
## [1,] 17.748120 16.9013012 16.5312375 16.3789524 16.3261470
## [2,]  1.019424  1.9845075  2.8279865  3.5568992  4.1941358
## [3,] -1.890519 -0.8468187 -0.3700636 -0.1522851 -0.0528054
## [4,]  1.019424  1.2272059  1.2662696  1.2742696  1.2759326
## > pred5[1,2:5]-pred5[1,1:4]
## [1] -0.8468187 -0.3700636 -0.1522851 -0.0528054
## > pred200=I1.predict(x,ar=fit.y$coef[1],ma=fit.y$coef[2],mu=fit.y$coef[3],origin=500,h=200,sigma2=fit.y$sigma2)
## > gamma=auto.cov(ar=fit.y$coef[1],ma=fit.y$coef[2],lag=200-1,sigma2=fit.y$sigma2)
## > sum(gamma*200:1)*2-200*gamma[1]
## [1] 1030.584
## > pred200[2,200]^2
## [1] 1028.585
