library(dplyr)
library(tidyr)


data.name <- "dpp"


dn <- paste0("mcmod",data.name) ## name of data file, e.g., mcmoddpp
data(list=dn)  ## load data
mcmod <- eval(parse(text=dn)) ## rename to mcmod
mh.file <- paste0("./nobuild/results/langMH_",data.name,".Rdata")
save.file <- paste0("./nobuild/results/sumMH_",data.name,".Rdata")


load(mh.file)
F <- melt(draws) %>%
  tidyr::separate(var, into=c("par","D1","D2"), sep="\\.",
                  remove=FALSE, fill="right")

Q <- group_by(F, var, par, v1, v2) %>%
  summarize(mean=mean(value),
            sd=sd(value),
            Q.025=quantile(value,.025),
            Q.05=quantile(value,.05),
            Q.25=quantile(value,.25),
            Q.50=quantile(value,.50),
            Q.75=quantile(value,.75),
            Q.95=quantile(value,.95),
            Q.975=quantile(value,.975))



