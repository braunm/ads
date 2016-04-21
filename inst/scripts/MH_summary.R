library(dplyr)
library(tidyr)
library(coda)
library(reshape)
library(ggplot2)
library(stringr)


geweke <- function(x,...) {
    R <- geweke.diag(x, ...)
    return(R$z)
}

data.name <- "dpp"

dn <- paste0("mcmod",data.name) ## name of data file, e.g., mcmoddpp
data(list=dn)  ## load data
mcmod <- eval(parse(text=dn)) ## rename to mcmod
mh.file <- paste0("./nobuild/results/langMH_",data.name,".Rdata")
save.file <- paste0("./nobuild/results/sumMH_",data.name,".Rdata")

start.iter <- 150000
iter_draw <-  200000

cat("Loading\n")
load(mh.file)
cat("Processing draws\n")
F <- melt(draws) %>%
  tidyr::separate(var, into=c("par","D1","D2"), sep="\\.",
                  remove=FALSE, fill="right")
cat("Summarizing draws\n")
Q <- filter(F, iter > start.iter & !is.na(value)) %>%
  group_by(var, par, D1, D2) %>%
  summarize(mean=mean(value),
            sd=sd(value),
            Z=mean/sd,
            Q.025=quantile(value,.025),
            Q.05=quantile(value,.05),
            Q.25=quantile(value,.25),
            Q.50=quantile(value,.50),
            Q.75=quantile(value,.75),
            Q.95=quantile(value,.95),
            Q.975=quantile(value,.975),
            P0=mean(value<0),
            n.draws=length(value),
            effSize=effectiveSize(value),
            geweke.z=geweke(value)) %>%
  ungroup()


#gew.logpost <- data_frame(iter=iter_draw, value=logpost) %>%
#  filter(iter > start.iter & !is.na(value)) %>%
#  select(value) %>%
  geweke(mcmc(logpost))

 gew <- dplyr::filter(F, iter>start.iter & !is.na(value)) %>%
   group_by(var, par, D1, D2) %>%
   summarize(geweke.z=geweke(value))

cat("Building plots\n")

trace_phi <- dplyr::filter(F,par=="phi" & iter>=start.iter) %>%
  ggplot(aes(x=iter,y=value)) %>%
  + geom_line(size=.1) %>%
  + geom_hline(yintercept=0,color="red") %>%
  + facet_grid(D1~D2, scales="free")


trace_theta12 <- dplyr::filter(F,str_detect(var,"theta12")) %>%
  ggplot(aes(x=iter,y=value)) %>%
  + geom_line(size=.1) %>%
  + facet_wrap(~var, scales="free")

#trace_logpost <- data_frame(iter=iter_draw, value=logpost) %>%
#  dplyr::filter(iter>=start.iter) %>%
#  ggplot(aes(x=iter, y=value)) %>%
#  + geom_line(size=.1)











