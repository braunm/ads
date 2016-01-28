



## Script to regenerate mcmod files
categories <- c("dpp","ptw","tti","ptw","lld")

T <- 226
N <- 42
fweek <- 1200
covv <- c("avprc")
covnv <- c("fracfnp","fracdnp","fracdist","numproducts")


for(category in categories) {
    cat("category: ",category,"\n")
    if(category=='lld') {
        ## liquid laundry detergents
        brands.to_keep<-c('TIDE','ALL','PUREX','GAIN',
                          'CHEER','WISK','AH','PL')  # 91.7% share
    }

    if(category=='tti'){
        ## toilet tissue
        brands.to_keep<-c('CHARMIN','QUILTEDNORTHERN',
                          'SCOTT','ANGELSOFT','PL') # (including Kleenex) 93.0% of share
    }

    if(category=='fti'){
        ## facial tissue
        brands.to_keep<-c('KLEENEX','PUFFS','PL') # 93.0% of share
    }

    if(category=='dpp'){
        ## disposable diapers
        brands.to_keep<-c('HUGGIES','PAMPERS','LUVS','PL')  # 98.2% of share
    }

    if(category=='ptw'){
        ## paper towels
        brands.to_keep<-c('BOUNTY','BRAWNY','SCOTT',
                          'VIVA','SPARKLE','PL') # 94.2% of share
    }


    mcmod <- mcmodf(data.name = category,
                    brands.to_keep = brands.to_keep,
                    covv = covv, covnv = covnv, T = T,
                    N = N, fweek = fweek, aggregated = FALSE)

    file.name <- paste0("mcmod",category)
    dn1 <- paste0(file.name,"<- mcmod")
    eval(parse(text=dn1))
    dn2 <- paste0("devtools::use_data(",file.name,", overwrite=TRUE)")
    eval(parse(text=dn2))

}

