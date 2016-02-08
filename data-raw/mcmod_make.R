## Script to regenerate mcmod files
categories <- c("dpp","ptw","fti","tti","lld")
# categories <- c("dpp")
T <- 202
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
brands.to_keep[['ptw']] 	<- c('BOUNTY','BRAWNY','SCOTT',
'VIVA','SPARKLE','PRIVATE LABEL') # 94.2% of share


for(category in categories) {
    cat("category: ",category,"\n")

    mcmod <- mcmodf(data.name = category,
                    brands.to_keep = brands.to_keep[[category]],
                    covv = covv, covnv = covnv, T = T,
                    N = N, fweek = fweek, aggregated = FALSE)

    file.name <- paste0("mcmod",category)
    dn1 <- paste0(file.name,"<- mcmod")
    eval(parse(text=dn1))
    dn2 <- paste0("devtools::use_data(",file.name,", overwrite=TRUE)")
    eval(parse(text=dn2))

}
