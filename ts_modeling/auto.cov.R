source("memory.R")


auto.cov=function(ar=0,ma=0,sigma2=1,lag){
  p=length(ar)
  q=length(ma)
  m=max(p,q)+1
  Phi=array(0,c(m,m))
  phi=rep(0,m)
  phi[1]=1
  phi[1:p+1]=-ar
  theta=rep(0,m)
  theta[1]=1
  theta[1:q+1]=ma
  psi=memory(ar,ma,lag)
  b=rep(0,m)
  
  Phi[1,]=phi
  for (i in 2:m){
    Phi[i,1:(m-i+1)] = phi[i:m]
    Phi[i,2:i] = Phi[i,2:i] + phi[(i-1):1]
  }
  for (i in 1:m){
    b[i]=sum(psi[1:(m-i+1)]*theta[i:m])
  }
  
  gamma=rep(0,lag+1)
  gamma[1:m]=solve(Phi)%*%b
  for (k in m:lag){
    gamma[k+1] = sum(ar*gamma[k:(k-p+1)])
  }
  return(gamma*sigma2)
}