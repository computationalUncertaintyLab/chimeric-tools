memory=function(ar=0,ma=0,lag){
  p=length(ar)
  q=length(ma)
  theta=rep(0,lag+1)
  theta[1]=1
  theta[2:(q+1)]=ma
  phi=rep(0,lag+1)
  phi[1]=1
  phi[2:(p+1)]=ar
  psi=rep(0,lag+1)
  psi[1]=1
  
  for (k in 1:lag){
    psi[1+k] = sum(phi[1+1:k]*psi[k:1]) + theta[1+k]
  }
  
  return(psi)
}