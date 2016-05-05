## Script to do summary statistics

library(data.table)
library(xtable)
library(MASS)


categories <- c("dpp","ptw","fti","tti","lld")
## categories <- c("ptw")
T <- 226
N <- 42
fweek <- 1200
covv <- c("avprc")
covnv <- c("fracfnp","fracdnp","fracdist","numproducts")

brands.to_keep <- list()
brands.to_keep[["lld"]] 	<- c('TIDE','ALL','PUREX','GAIN',
'CHEER','WISK','ARM & HAMMER','PRIVATE LABEL')  # 91.7% share
brands.to_keep[["tti"]] 	<- c('CHARMIN','QUILTED NORTHERN',
'SCOTT','PRIVATE LABEL') # (including Kleenex) 93.0% of share
brands.to_keep[["fti"]] 	<- c('KLEENEX','PUFFS','PRIVATE LABEL') # 93.0% of share

brands.to_keep[["dpp"]]		<- c('HUGGIES','PAMPERS','LUVS','PRIVATE LABEL')  # 98.2% of share
brands.to_keep[['ptw']] 	<- c('BOUNTY','BRAWNY','SCOTT','VIVA','SPARKLE','PRIVATE LABEL') # 94.2% of share

## summary data


## sales, distribution, revenues etc.
#salesdf <- data.frame(category = as.character(), Nbrands = as.integer(), Revenue = as.numeric(), price = as.numeric(), fracdnp = as.numeric(), fracfnp = as.numeric(), fracdist = as.numeric())
salesdf <- NULL

for(category in categories) {
    DT <- mcmodf(data.name = category,
    brands.to_keep = brands.to_keep[[category]],
    covv = covv, covnv = covnv, T = T,
    N = N, fweek = fweek, aggregated = FALSE, summary.only = TRUE)

    salesdf <- rbind(salesdf,DT[,list(category,Nbrand = uniqueN(brand), revenue = sum(dollars)/uniqueN(week), price = sum(dollars)/sum(volume), fracdnp=mean(fracdnp), fracfnp = mean(fracfnp), fracdist=mean(fracdist), numproducts=mean(numproducts), avgnatadspend = mean(natdols,na.rm=TRUE))])
}

## print LaTeX table for marketing mix
print(xtable(salesdf, digits = c(0,0, 1,1,rep(2,4),0,0), caption="Sales and marketing mix statistics for all 42 cities in sample, and for 226 weeks of data. Categories being lld = liquid laundry detergents, tti = toilet tissue, fti = facial tissue, dpp = disposable diapers, ptw = paper towels.  Revenues are weekly market revenues and represents a sample of stores and is aggregated across 42 cities. Price represents average price per volume. avgnatadspend is weekly expenditure on national advertising by all brands, fracdnp/fnp is the fraction of UPCs available that were on display/feature, fracdist = average distribution (number of outlets), numproducts is the average number of SKUs per brand."), include.rownames=FALSE)



## categories <- c("ptw")
categories <- c("dpp","ptw","fti","tti","lld")

## creatives across categories
acats <- NULL
for(category in categories) {
    cat("category: ",category,"\n")
    a <- getcreativesummary(category, brands.to_keep[[category]])
    acats <- rbind(acats,a[])
}


adsdf <- acats[ ,list(avgdolpa = sum(totalspent/1.0e6/T*52/nbrands), totdol = mean(totalspent/1.0e6), fwpct = mean(fracspent.first.week,na.rm=TRUE), minutes = mean(time.mins), Nc = uniqueN(cID)/min(nbrands*T/52), ml = mean(time1), p25l = quantile(time1, probs=0.25), p50l = quantile(time1, probs=0.5), p75l = quantile(time1, probs=0.75), p95l = quantile(time1, probs=0.95)), by = categoryname]

print(xtable(adsdf, digits = c(0, 0, 2, 2, 2, 0, 1, rep(0,5) ), caption = "Advertising and creative summary statistics per category, with 
avgdolpa = average annual advertising expenditure per brand, totdol = total amount spent on new creative over its lifetime, fwpct = fraction of a brands ad budget spent on new creative in first week of showing, minutes = total minutes of ad exposure per new creative, Nc = number of new creatives per brand per year, ml = average length of lifetime for creative, and pXXl being the (25,50, 75 and 95) percentiles for length of time creatives are used."), include.rownames=FALSE)


##----plots

## length of time
par(mfrow=c(3,2))

h <- acats[,hist(time1, main="All categories", ylim = c(0,500), xlab="Time in weeks")]
## Weibull parameters
.wp <- acats[,fitdistr(1+time1, 'weibull')]$estimate

xfit <- acats[,seq(min(time1),max(time1),length=40)]
yfit <- dweibull(xfit, shape= .wp[1], scale= .wp[2])
yfit <- yfit * diff(h$mids[1:2])*acats[,.N]
lines(xfit, yfit, col= "blue", lwd=2)
text(150, 400, pos = 2, paste0("Weibull shape = ",round(.wp[1],2)))
text(150, 340, pos = 2, paste0("Weibull scale = ",round(.wp[2],2)))

## across categories
for(category in categories) {
    .acat <- acats[categoryname==category,]
    h <- .acat[,hist(time1, main=paste0("Category: ", category), ylim = c(0,200), xlab="Time in weeks")]
    ## Weibull parameters
    .wp <- .acat[,fitdistr(1+time1, 'weibull')]$estimate

    xfit <- .acat[,seq(min(time1),max(time1),length=40)]
    yfit <- dweibull(xfit, shape= .wp[1], scale= .wp[2])
    yfit <- yfit * diff(h$mids[1:2])*.acat[,.N]
    lines(xfit, yfit, col= "blue", lwd=2)
    text(max(xfit), 125, pos = 2, paste0("Weibull shape = ",round(.wp[1],2)))
    text(max(xfit), 100, pos = 2, paste0("Weibull scale = ",round(.wp[2],2)))
}


## fraction spent per week
par(mfrow=c(3,2))

h <- acats[,hist(fracspent.first.week, main="All categories", ylim = c(0,400), xlab="Fraction of weekly spend")]

## across categories
for(category in categories) {
    .acat <- acats[categoryname==category,]
    h <- .acat[,hist(fracspent.first.week, main=paste0("Category: ", category), ylim = c(0,120), xlab="Fraction of weekly spend")]

}


## fraction spent per week
par(mfrow=c(3,2))

h <- acats[,hist(totalspent/1.0e6, main="All categories", ylim = c(0,600), xlab="Total spent ($m)")]

## across categories
for(category in categories) {
    .acat <- acats[categoryname==category,]
    h <- .acat[,hist(totalspent/1.0e6, main=paste0("Category: ", category), ylim = c(0,200), xlab="Total spent ($m)")]
    }


## time series for creative mix
data.name <- "dpp"
data.is.sim <- FALSE

dn <- paste0("mcmod",data.name) ## name of data file, e.g., mcmoddpp
data(list=dn)  ## load data
mcmod <- eval(parse(text=dn)) ## rename to mcmod

par(mfrow=c(mcmod$dimensions$R,1))
cm <- array(unlist(mcmoddpp$CM), dim = c(mcmod$dimensions$Jb,mcmod$dimensions$R, mcmod$dimensions$T))

matplot(t(cm[,1,]), type='l',main="Creative mix: novelty", col=c("black","red","green"),xlab="Average creatives", ylab="week")
abline(h=colMeans(t(cm[,1,])),col=c("black","red","green"))
legend(0,85,brands.to_keep[["dpp"]][1:mcmod$dimensions$Jb],col=c("black","red","green"),lty=1:mcmod$dimensions$Jb, cex=0.7)

matplot(t(cm[,2,]), type='l',main="Creative mix: variety", xlab="Average age in weeks", ylab="week")
abline(h=colMeans(t(cm[,2,])),col=c("black","red","green"))

matplot(t(cm[,3,]), type='l',main="Creative mix: concentration", xlab="Concentration index", ylab="week")
abline(h=colMeans(t(cm[,3,])),col=c("black","red","green"))

## end of plots


## this creates the OLS for time spent on to some category specific/creative specific variables

tm1 <- lm(log(1+time1) ~ log(nproperty) + log(nprograms) + log(nprogtype) + categoryname + fracspent.first.week + dols.first.week, data=acats)






