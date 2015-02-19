rm(list=ls())
gc()

mod.name <- "hdlm"
data.name <- "ptw"

data.file <- paste0("data/mcmod",data.name,".RData")
save.file <- paste0("inst/results/",mod.name,"_",data.name,"_regress.Rdata")

load(data.file)

Y1 <- mcmod$Y[[1]]
X1 <- mcmod$X[[1]]
N <- mcmod$dimensions$N
P <- mcmod$dimensions$P
J <- mcmod$dimensions$J
K <- mcmod$dimensions$K
F <- matrix(0,N,1+J)

for (i in 1:N) {
    r_ <- ((i-1)*(1+J)+1):(i*(1+J))
    F[i,] <- mcmod$F1[[1]][r_, i]
}

D <- list(length=J)
reg <- list(length=J)


for (i in 1:J) {
    D[[i]] <- data.frame(Y=Y1[,i],
                         p1=F[,2], p2=F[,3], p3=F[,4],
                         PF=X1[,i],
                         PD=X1[,i+J],
                         SAd=X1[,i+2*J])        
    reg[[i]] <- lm(Y~p1+p2+p3+PF+PD+SAd, data=D[[i]])
}
