## Script to regenerate mcmod files
## set path to IRI root
library(data.table)
library(lubridate)

rm(list=ls())
gc()

categories <- c("dpp","ptw","fti","tti","lld")
#categories <- "ber"

reload.data <- TRUE
#categories <- c("lld")
T <- 226
N <- 50
fweek <- 1200
max.distance = 0.1
select.markets.from.file <- FALSE          ## to get market list from market_list.R

covv <- c("avgprc")
#covnv <- c("fracfnp","fracdnp","fracdist","numproducts")
covnv <- c("feature","display","priceoff","fracwdist","numproducts")

brands.to_keep <- list()
brands.to_keep[["lld"]] 	<- c('TIDE','ALL','GAIN',
'CHEER','PUREX','WISK','ARM & HAMMER','PRIVATE LABEL')  # 91.7% share
brands.to_keep[["tti"]] 	<- c('KLEENEX','CHARMIN','QUILTED NORTHERN',
'SCOTT','PRIVATE LABEL') # (including Kleenex) 93.0% of share
brands.to_keep[["fti"]] 	<- c('KLEENEX','PUFFS','PRIVATE LABEL') # 93.0% of share

brands.to_keep[["dpp"]]		<- c('HUGGIES','PAMPERS','LUVS','PRIVATE LABEL')  # 98.2% of share
brands.to_keep[["ptw"]] 	<- c('BOUNTY','BRAWNY','SCOTT','SPARKLE','KlEENEX VIVA','PRIVATE LABEL') # 94.2% of share
brands.to_keep[['ber']] 	<- c('BUDWEISER','HEINEKEN') # xx% of share but more stable

for(data.name in categories) {
    brandlist <- brands.to_keep[[data.name]]
    if(reload.data) {
        cat("Reading from IRI and TNS flat files.\n")
        source("./data-raw/read.data.R")
            
    }
    cat("Processing data for category: ", data.name,"\n")

    mcmod <- mcmodf(data.name,
                    brands.to_keep = brands.to_keep[[data.name]],
                    covv = covv, covnv = covnv, T = T,
                    N = N, fweek = fweek, aggregated = FALSE,max.distance=max.distance, ads.from.tns = TRUE, minadv = 0.25e6, use.iri = TRUE)
    
    file.name <- paste0("mcmod",data.name)
    dn1 <- paste0(file.name,"<- mcmod")
    eval(parse(text=dn1))
    dn2 <- paste0("devtools::use_data(",file.name,", overwrite=TRUE)")
    eval(parse(text=dn2))

}
