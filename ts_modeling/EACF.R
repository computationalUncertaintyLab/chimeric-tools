## Courtesy of Professor Ruey Tsay

"EACF" <- function(x,p=6,q=8){
  x=as.matrix(x)
  pq=p+q
  x1=x-mean(x)
  n=dim(x1)
  coef=matrix(rep(0,pq*pq),pq,pq)
  for (i in 1:pq){
    md=arma(x1,order=c(i,0), include.intercept=F)
    for (j in 1:i){
      coef[i,j]=md$coef[j]
    }
  }
  
  eacf=matrix(rep(0,p*q),p,q)
  m1=acf(x1,lag.max=q,plot=F)
  for (j in 1:q){
    eacf[1,j]=m1$acf[j+1]
  }
  
  
  for (j in 1:q) {
    tmp=coef
    for (i in 1:(pq-j)){
      for (ii in 1:i){
        if (ii == 1) coef[i,ii]= tmp[i+1,ii]+tmp[i+1,i+1]/tmp[i,i]
        if (ii >=2) coef[i,ii]=tmp[i+1,ii]-tmp[i+1,i+1]*tmp[i,ii-1]/tmp[i,i]
      }
    }
    
    for (i in 2:p){
      y=matrix(rep(0,n[1]),n[1],1)
      for (it in i:n[1]) {
        w=x1[it,1]
        for (ii in 1:(i-1)){
          w=w-coef[(i-1),ii]*x1[it-ii,1]
        }
        y[it,1]=w
      }
      m1=acf(y,lag.max=q,plot=F)
      eacf[i,j]=m1$acf[j+1]
    }
  }
  
  seacf=matrix(rep(0,p*q),p,q)
  sd = 2/sqrt(n[1])
  for (i in 1:p){
    for (j in 1:q){
      if (abs(eacf[i,j]) >= sd) {seacf[i,j]=2}
    }
  }
  
  print('EACF table')
  print(eacf)
  print(' ')
  print('Simplified EACF: 2 denotes significance')
  
  print(seacf)
  return(list(eacf=eacf,seacf=seacf))
}