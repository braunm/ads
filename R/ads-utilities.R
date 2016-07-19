#' rmvMN
#' Draw samples from a matrix normal using Cholesky decomposition.
#'
#' @param ndraws Number of draws
#' @param M matrix mean (must by dimension nrow(S) x nrow(C))
#' @param C covariance matrix representing columns
#' @param S covariance matrix representing rows
#' @return ndraws matrices each of dimension S x C of random normal distribution
#' @examples
#' rmvMN(1, C, S)
rmvMN <- function(ndraws, M = rep(0, nrow(S) * ncol(C)), C, S) {
    ## set.seed(153)
    L <- chol(S) %x% chol(C)
    z <- rnorm(length(M))
    if (length(M) == 1) {
        res <- matrix(t(L) %*% z, nrow = nrow(C), ncol = ncol(S))
    }  else {
        res <- matrix(as.vector(M) + t(L) %*% z, nrow = nrow(C), ncol = ncol(S))
    }
    return(res)
}

#' mcmodf
#' Create data files for analysis from raw IRI and TNS data files.
#'
#' @param data.name Three letter category acronym (being one of dpp, fti, lld, ptw, tti)
#' @param brands.to_keep List of brands corresponding to the Y matrix (being volume sales)
#' @param covv Character vector for time varying covariates, being one of 'avprc', 'lavprc', and others
#' @param T Integer value for number of weeks to include from and inclusive of fweek
#' @param N Integer value for number of markets to include (constructs from market_list being an alphabetic list)
#' @param fweek Integer value for first week to start analysis
#' @param aggregated (Defaults to FALSE) Boolean flag to indicate generation of aggregated (all US) versus city level data
#' @param brands.advertised If NULL, the code just tries to guess from the dataset provided for ads.
#' @param ads.from.tns TRUE or FALSE, if TRUE then extracts data for advertising from a separate TNS data set
#' @return mcmod object being a list with the following elements: list(dimensions=dimensions,Y = Y,CM = CMl, E = El, A = Al, X = Xl, F1 = F1l, F2 = F2)
#' @examples
#' mcmod() - Runs default with "dpp"
#' mcmod(data.name = "dpp", brands.to_keep = c('HUGGIES','PAMPERS','LUVS','PL'), covv = c("avprc"), covnv = c("fracfnp","fracdnp","fracdist","numproducts"), T = 226,  N = 42, fweek = 1200, aggregated = FALSE)
mcmodf <- function(data.name = "dpp", brands.to_keep = c('HUGGIES','PAMPERS','LUVS','PRIVATE LABEL'), ads.from.tns = FALSE, use.iri = FALSE, brands.advertised = NULL, covv = c("avprc"), covnv = c("fracfnp","fracdnp","fracdist","numproducts"), T = 226,  N = 42, fweek = 1200, aggregated = FALSE, max.distance = 0.2, minadv = 0.25e6, summary.only=FALSE) {

    J = length(brands.to_keep)
    P <- J * length(covv)          # total number of time varying covariates at city level
    # non time varying componentns
    K <- J * length(covnv)

    category <- data.name

    ###### read in data file required and make any transforms needed
    if(use.iri) {
       DT <- readRDS(paste0("nobuild/data-raw/",data.name,"iri.RDS"))
       setkey(DT,market_name,week,brand)
    } else {
        dfname <- sprintf('nobuild/data-raw/%siritns.txt',data.name)
    
        ## Trim characters, just in case
        DT <- fread(dfname)
        DT <- trim_characters(DT)
    }

    market_list <- unique(DT$market_name)

    # keep only brands focused on here, and subset of weeks and markets
    DT <- DT[brand %in% brands.to_keep & week < fweek + T & week >= fweek & market_name %in% market_list[1:N],]

    if(summary.only) return(DT)
    
    # only do the rest if balanced
    if(any(dcast.data.table(DT, week ~ market_name, value.var="volume", subset = .(week < fweek+T & week >= fweek & market_name %in% market_list[1:N]), fun = length) < J)) {
        warning("Warning: Cannot do analysis on brands with unbalanced data: reduce brands in brands.to_keep, or choose a subset of markets. Step unable to complete for analysis, but ok for summary data (use DT)")
    } else {

        # create outcome variable (Y)
        Y <- structure(rep(0,T*N*J), dim=c(T,N,J))
        for(j in 1:J) {
            .a <- dcast.data.table(DT, week ~ market_name, value.var = "volume", subset = .(week >= fweek & brand %in% brands.to_keep[j]), fill=0, fun = min)
            .a[,week := NULL]
            Y[,,j] <- simplify2array(.a)
        }
        rm(.a)

        # this next one just pulls the ads directly from a TNS file, and does not care about order so much
        # TNS file having rows by IRI week
        if(ads.from.tns) {
            TNSdt <- readRDS(paste0("nobuild/data-raw/", data.name, "TNS.RDS"))
            .a <- dcast(TNSdt, weekID ~ brand, value.var="dols", subset= .(media %in% c("Cable TV", "SLN TV", "Network TV", "Syndication","Spot TV") & weekID >= fweek), fill=0, fun=sum)
            
            brands_advertised <- names(sort(colSums(.a)[-1],decreasing=TRUE))
            .ai <- which(colSums(.a[,brands_advertised,with=F]) > minadv)
            XAdv <- .a[,brands_advertised[.ai],with=F]
        } else {
            # create advertising variable (XAdv), national only here (using mean, therefore)
            DT[is.na(natdols),natdols:=0]
            XAdv <- dcast.data.table(DT, week ~ brand, value.var = "natdols", subset = .(week >= fweek), fill = 0, fun = mean, na.rm = TRUE)
            XAdv <- XAdv[,brands.to_keep,with=FALSE]			# reorder columns to correspond to list
        }
        
        # create covariates from list, X being not time varying, and X2 being time varying

        X1 <- structure(rep(0,T*N*J*length(covnv)), dim=c(T,N,J*length(covnv)))
        for(mkt in 1:N) {
            for(j in 1:J) {
                .a <- dcast.data.table(DT, week ~ market_name, value.var = covnv, subset = .(week >= fweek & market_name == market_list[mkt] & brand == brands.to_keep[j]), fun = mean, na.rm = TRUE)
                .a[,week := NULL]
                if(j ==1) .xcov <- simplify2array(.a) else .xcov <- cbind(.xcov, simplify2array(.a))
            }
            X1[,mkt,] <- .xcov
        }
        rm(.a,.xcov)

        X2 <- structure(rep(0,T*N*J*length(covv)), dim=c(T,N,J*length(covv)))
        for(mkt in 1:N) {
            for(j in 1:J) {
                .a <- dcast.data.table(DT, week ~ market_name, value.var = covv, subset = .(week >= fweek & market_name == market_list[mkt] & brand == brands.to_keep[j]), fun = mean, na.rm = TRUE)
                .a[,week := NULL]
                if(j ==1) .xcov <- simplify2array(.a) else .xcov <- cbind(.xcov, simplify2array(.a))
            }
            X2[,mkt,] <- .xcov
        }
        rm(.a,.xcov)
    }
    

    ####### work on advertising data
    # first get brands advertising over this period

    A <- simplify2array(XAdv)[1:T,]
    brands_advertised <- brands_advertised[as.numeric(which(colSums(A)>0))]
    A <- A[,brands_advertised]
    colnames(A) <- make.names(colnames(A),unique=TRUE)                 # fix column names with dots, not spaces
    
    Jb <- length(brands_advertised)

    ## for this subset, which brands launched new creatives in that time frame?
    # call creatives function for Jb brands

    .a <- getcreatives(category, brands_advertised, T, max.distance=max.distance, ads.from.tns,TNSdt)

    ownadnnc <- .a$ownadnnc
    ownadnnc.fracspent <- .a$ownadnnc.fracspent
    ownadnnc.fracspent.l1 <- .a$ownadnnc.fracspent.l1
    
    #load(paste(codepath,"/",category,"creatives.RData",sep=""))
    s.nnc <- as.data.frame(ownadnnc[weekID >= fweek & weekID < fweek+T,brands_advertised,with=FALSE])
    s.nnc$weekID <- NULL

    ## same but multiplied by fraction of ad budget that week by that brand
    s.nnc.fracspent <- as.data.frame(ownadnnc.fracspent[weekID >= fweek & weekID < fweek+T,brands_advertised,with=FALSE])
    s.nnc.fracspent$weekID <- NULL

    ## same but multiplied by fraction of ad budget that week by that brand
    s.nnc.fracspent.l1 <- as.data.frame(ownadnnc.fracspent.l1[weekID >= fweek & weekID < fweek+T,brands_advertised,with=FALSE])
    s.nnc.fracspent.l1$weekID <- NULL

    .brands_changed <- which(colSums(s.nnc)>0)
    brands_nnc <- brands_advertised[.brands_changed]

    ## Data for creatives - note if any adv = 0 then columns removed
    ## 2/2/2015 - now E is same dimension as ads (Jb) but JbE measures the number of brands with observed changes so JbE could be less than Jb
    ## E <- s.nnc[1:T,.brands_changed]
    E <- s.nnc[1:T,]
    Ef <- s.nnc.fracspent[1:T,]
    Efl1 <- s.nnc.fracspent.l1[1:T,]
    JbE <- length(brands_nnc)
    JbEv <- match(brands_nnc, brands_advertised)			# pointer to brands.adv, which ones changed

    R <- length(.a$creativemix)

    ####################### set up structure for mcmod

    #### set up dependent variables here
    Y <- log(Y[1:T,1:N,])

    # F1
    F1 <- array(0, dim = c(T, N, N * (1+P)))
    for(t in 1:T) for(mkt in 1:N) F1[t, mkt, (mkt-1) * (1+P) + 1:(1+P)] <- c(1,log(X2[t, mkt, ]))

    # F2
    b1 <- Matrix(c(1, rep(0, Jb + P)), nrow = 1, ncol = 1 + Jb + P)
    b2 <- Matrix(0, nrow = P, ncol = 1 + Jb)
    b3 <- Diagonal(P)
    B <- rBind(b1, cBind(b2, b3))

    F2 <- list()
    for (t in 1:T) {
        F2[[t]] <- kronecker(Matrix(rep(1, N), nrow = N, ncol = 1), B)
        F2[[t]] <- t(as(F2[[t]], "dgCMatrix"))
    }

    # create final mcmod components
    Al <- El <- Efl <- Efl1l <- CMl <- CMdl <- Xl <- F1l <- Yl <- list()
    for (t in 1:T) {
        El[[t]] <- matrix(as.numeric(E[t,]),nr=Jb,nc=1)
        Efl[[t]] <- matrix(as.numeric(Ef[t,]),nr=Jb, nc=1)
        Efl1l[[t]] <- matrix(as.numeric(Efl1[t,]),nr=Jb, nc=1)
        CMl[[t]] <- .a$creativemix[[1]][weekID == fweek+(t-1),brands_advertised,with=FALSE]/52
        .cd <- .a$creativemix[[1]][weekID == fweek+(t-2),brands_advertised,with=FALSE]
        for(r in 2:R) {
            CMl[[t]] <- rbind(CMl[[t]], .a$creativemix[[r]][weekID == fweek+(t-1),brands_advertised, with=FALSE]/10)
            .cd <- rbind(.cd, .a$creativemix[[r]][weekID == fweek+(t-2),brands_advertised, with=FALSE])
        }
        
        CMl[[t]] <- t(CMl[[t]])
        CMdl[[t]] <- CMl[[t]] - t(.cd) ## differenced version
        
   ##     names(El[[t]]) <- colnames(E)
        Al[[t]] <- as.numeric(A[t, ])           # to be sure
        Xl[[t]] <- X1[t,,]
        F1l[[t]] <- t(as(F1[t,,], "dgCMatrix"))
        Yl[[t]] <- Y[t,,]
    }

    dimensions <- list(N = N, T= T, J=J, R = R, Jb = Jb, JbE = JbE, K = ncol(Xl[[1]]), P = P, brands_advertised = brands_advertised, brands_sold = brands.to_keep)
    mcmod <- list(dimensions=dimensions,Y = Yl, CM = CMl, E = El, Ef = Efl, Efl1 = Efl1l, A = Al, X = Xl, F1 = F1l, F2 = F2)
    return(mcmod)
}

##' Generate creatives renewal,replacement,addition. 
##'
##' Builds a matrix of values for creatives. Currently only supports integer values indicating
##' the number of new creatives added.
##'
##' @param category Three letter category acronym (being one of dpp, fti, lld, ptw, tti)
##' @param brands.adv Which brands advertised
##' @param fweek Integer value for first week to start analysis
##' @param T Integer value for number of weeks to include from and inclusive of fweek
##' @param max.distance A value passed in that uses Levenshtein distance in agrep
##' @return T x Jb matrix with integer corresponding to number of new creatives added that week. (Note, removed make.binary flag so nnc is always integer for number of creatives introduced.)
##' @examples
##' getcreatives(category, brands.adv, Jb, fweek, T, make.binary=F) - called from within function mcmod()
getcreatives <- function(category, brands.adv, T, max.distance=0.2, ads.from.tns, DT) {

    Jb = length(brands.adv)

    if(ads.from.tns) {
        creatives <- DT
        # setnames(creatives,"week","weekID")
    } else {
        cf <- paste0("nobuild/data-raw/",category,"creatives.txt")
        ## convert brand names
        creatives <- fread(cf,col.names= c('brand','program','progtype','tvcreative','property','media','avg30','avg30d','dols','sec','dtime','firstdateshown','weekID','firstweekshown'))
        }
    

    ## Trim fields here just in case
    creatives <- trim_characters(creatives)
    # for(j in 1:Jb) creatives[agrep(brands.adv[j],brand,ignore.case=TRUE), brand:= brands.adv[j]]
    # creatives <- creatives[brand %in% brands.adv,]			# delete any brands not advertised in list

    # creatives[,ENEWS:=0]
    #### now generic code

    ## group creatives together and create a new creativeID (cID) for each
    ## Note that "other" creatives are assigned an ID of 5000 and all grouped together
    
    #### now generic code
    ## collapse
    # this sets creativeID

    setcreativeID(creatives,max.distance=max.distance)
    creatives <- creatives[brand %in% brands.adv,]           # remove brands excluded
    # count new ads (just use different labels, perhaps can improve on this with distance)
    #    creatives[weekID==cIDfweek, nnc := uniqueN(cID),by=c("weekID","brand")]
    #    creatives[is.na(nnc), nnc := 0]
    
    # if(make.binary) creatives[nnc>0,nnc:=1]

    # collapse to get total new creatives
    #.nnc <- creatives[,max(nnc),by=c("weekID","brand")]

    #ownadnnc <- merge(creatives[,.N,by=weekID], creatives[brand == brands.adv[1],max(nnc),by=c("weekID")],by="weekID",all.x=TRUE,all.y=F)
    #setnames(ownadnnc,"V1",brands.adv[1])
    #for(b in 2:Jb) {
    #    ownadnnc <- merge(ownadnnc, creatives[brand == brands.adv[b],max(nnc),by=c("weekID")],by="weekID",all.x=TRUE,all.y=F)
    #    setnames(ownadnnc,"V1", brands.adv[b])
    #}
    #ownadnnc <- ownadnnc[,N := NULL]

    return(getcreativemix(creatives))

#allweeks <- data.table(weekID=min(ownadnnc$weekID):max(ownadnnc$weekID))
#setkey(allweeks,weekID)
#   setkey(ownadnnc,weekID)

#    ownadnnc <- merge(allweeks,ownadnnc,all.x=TRUE)
#    ownadnnc[is.na(ownadnnc)] <- 0
#    return(ownadnnc)
}

##' Trim white space.
##'
##' Trim characters recursively in a data table.
##'
##' @param DT Data table name
##' @return A new data table with trimmed characters
##' @examples
##' b <- trim_characters(a)
trim_characters = function(DT) {
    df <- data.frame(DT)
    for (i in names(df)) if(is.character(df[,i])) df[,i] <- str_trim(df[,i])
    return(data.table(df))
}


##' Set creative IDs
##'
##' Takes a data table of creatives and assigns each unique creative with its own ID, and adds first week aired.
##'
##' @param DT A data table
##' @param Jb Number of brands advertised
##' @param max.distance Levenshtein's distance
##' @param brands.adv Which brands are advertised
##' @return data table with same number of rows as original, but with cID included (and first week)
setcreativeID <- function(DT,Jb=DT[,uniqueN(brand)], max.distance=0.2, brands.adv=DT[,unique(brand)]) {

    DT[,cID := (Jb+1)*1000]
    for(j in 1:Jb){
        li<-list()
        .c <- DT[brand==brands.adv[j],unique(tvcreative)]
        li[[1]] <- clist <- agrep(.c[1],.c,max.distance=max.distance)
        mc <- 1
        for(m in 2:length(.c)) if(!any(clist==m)) {
            mc<-mc+1
            li[[mc]]<-agrep(.c[m],.c,max.distance=max.distance)	# max.distance is measuring how difft two creative descriptors are
            clist<-c(clist,li[[mc]])
        }

        for(m in 1:length(li)) DT[tvcreative %in% .c[li[[m]]] & toupper(brand)==toupper(brands.adv[j]),cID:= j * 1000 + m]
    }

    ##
    DT[,cIDfweek:=min(firstweekshown),by=cID]
}


##' Get creative data for a category
##'
##' Creatives a data table from raw data on creatives
##'
##' @param category Three letter acronym for category (e.g. "dpp")
##' @param brands.to_keep Character vector of brands that advertise
##' @param max.distance Levenshtein's distance
##' @return A list with a data table for creatives, Jb being the maximum brands, and a brand list for brands advertised
getcreativedata <- function(category, brands.to_keep, max.distance = 0.2){
    
    cf <- paste0("~/Documents/ads/nobuild/data-raw/",category,"creatives.txt")
    
    ## convert brand names
    creatives <- fread(cf,col.names= c('brand','program','progtype','tvcreative','property','media','avg30','avg30d','dols','sec','dtime','firstdateshown','weekID','fweek'))
    creatives <- trim_characters(creatives)
    creatives[,brand:=toupper(brand)]
    creatives <- creatives[brand %in% brands.to_keep,]
    
    creatives <- creatives[!like(toupper(tvcreative),"CREATIVE UNKNOWN"),]
    brands.adv <- creatives[,unique(brand)]
    
    Jb = length(brands.adv)
    
    creatives <- creatives[brand %in% brands.adv,]			# delete any brands not advertised in list
    
    ## collapse
    setcreativeID(creatives)
    
    return(list(creatives = creatives,Jb = Jb,brands.adv = brands.adv))
}


##' Make creative mix
##'
##' Returns a matrix of weekly observations including whether new ads were released, or
##' some metric of average age.
##'
##' @param creatives The creative file itself
##' @return A list with a list of metrics for creatives, each element being a T x Jb,
##' @return data table, with Jb being the maximum brands on columns
getcreativemix <- function(creatives){

    ## Average age per cID, number of new creatives, average number of creatives
    creatives[,cIDavgage:=weighted.mean(weekID-cIDfweek,dols),by=c("brand","weekID")]
    ## Total spent per brand for a given week
    creatives[,totalspent := sum(dols),by=c("brand","weekID")]
    creatives[,cIDfracspent := sum(dols/totalspent),by=c("cID","brand","weekID")]
    
    ## when a creative was introduced, and how many per week were introduced
    .a <- creatives[cIDfweek == weekID,list(nnc=uniqueN(cID)),by=c("brand","weekID")]
    setkey(.a,brand,weekID)
    setkey(creatives,brand,weekID)
    .b <- .a[creatives][is.na(nnc),nnc:=0]
    setkey(.b, weekID)
    .c <- .b[.(min(weekID):max(weekID)), roll=TRUE] # fill out just in case
    ##    if(any(.c[,is.na(nnc)])) stop("This step produces NAs for nnc, you must debug")

    ## 1. Integer count of number of new creatives from beginning of series
    ownadnnc <- dcast(.c[,list(nnc = max(nnc)),by=c("brand","weekID")], weekID ~ brand, value.var = "nnc", fun=max, fill=0)

    ## The following are metrics across brands, for each time period, going in as columns to a
    ## an object (CM, being Jb x R), currently R is set at three. This object is used instead
    ## of E in the H matrix.

    ## 2. Creative mix element 1: Novelty being expenditure weighted average age
    creativemix <- list()
    .a <- data.table(na.approx(dcast(creatives, weekID ~ brand, value.var = "cIDavgage", fun=max, fill=NA)))
    setkey(.a, weekID)
    ## the following step ensures that weekID is continuous (with values being interpolated)
    creativemix[[1]] <- .a[.(min(weekID):max(weekID)), roll=TRUE]

    ## 3. Creative mix element 2: being number of creative ads being shown, weighted by spend
    .a <- data.table(na.approx(dcast(creatives, weekID ~ brand, value.var = "cID", fun = uniqueN, fill = NA)))
    setkey(.a, weekID)
    creativemix[[2]] <- .a[.(min(weekID):max(weekID)), roll=TRUE]
    
    ## 4. Creative mix element 3: Concentration (similar to Herfindahl index) being total spent on advertising for
    ##    each brand, and share by each creative for that week
    .c <- creatives[,list(fracspent = sum(ifelse(totalspent>0, dols/totalspent, 0))), by=c("cID","brand","weekID")]
    .c[,sum_fracspent_squared := sum(fracspent^2),by=c("brand","weekID")]
    
    .a <- dcast(.c, weekID ~ brand, value.var = "sum_fracspent_squared", fun=max, fill = 0)
    setkey(.a, weekID)
    creativemix[[3]] <- .a[.(min(weekID):max(weekID)), roll=TRUE]
    
    ## This component is to measure ownadnnc weighted by expenditure share, and E.lag variables
    ##  This is for the share weighted values of new cIDs
 
    .a <- creatives[cIDfweek == weekID,list(cIDfracspent = min(cIDfracspent)),by=c("brand","cID","weekID")] ## collapse first
    .a <- .a[,list(nnc.fracspent = sum(cIDfracspent)),by=c("brand","weekID")] ## now collapse by brand
    setkey(.a,brand,weekID)
    setkey(creatives,brand,weekID)
    .b <- .a[creatives][is.na(nnc.fracspent),nnc.fracspent := 0]
    setkey(.b, weekID)
    .c <- .b[.(min(weekID):max(weekID)), roll=TRUE] # fill out just in case

    ownadnnc.fracspent <- dcast(.c[,list(nnc.fracspent = max(nnc.fracspent)),by=c("brand","weekID")], weekID ~ brand, value.var = "nnc.fracspent", fun=max, fill=0)

    ## exactly the same, but lagged one week
    .a <- creatives[weekID == (cIDfweek+1),list(cIDfracspent.l1 = min(cIDfracspent)),by=c("brand","cID","weekID")] ## collapse first
    .a <- .a[,list(nnc.fracspent.l1 = sum(cIDfracspent.l1)),by=c("brand","weekID")] ## now collapse by brand
    setkey(.a,brand,weekID)
    setkey(creatives,brand,weekID)
    .b <- .a[creatives][is.na(nnc.fracspent.l1),nnc.fracspent.l1 := 0]
    setkey(.b, weekID)
    .c <- .b[.(min(weekID):max(weekID)), roll=TRUE] # fill out just in case

    ownadnnc.fracspent.l1 <- dcast(.c[,list(nnc.fracspent.l1 = max(nnc.fracspent.l1)),by=c("brand","weekID")], weekID ~ brand, value.var = "nnc.fracspent.l1", fun=max, fill=0)
    ## done, now returning what we came up with
    
    return(list(ownadnnc = ownadnnc, ownadnnc.fracspent = ownadnnc.fracspent, ownadnnc.fracspent.l1 = ownadnnc.fracspent.l1, creativemix = creativemix))
}



##' Make summary file for creatives
##'
##' Returns a number of weekly metrics for a given creative file
##' Script not yet complete. Ignores fweek.
##'
##' @return Something
getcreativesummary <- function(category, brands){
    
    cf <- paste0("./nobuild/data-raw/",category,"creatives.txt")
    
    ## convert brand names
    ## Note that firstdateshown/firstweekshown refers to original ad, which are later collapsed
    creatives <- fread(cf,col.names= c('brand','program','progtype','tvcreative','property','media','avg30','avg30d','dols','sec','dtime','firstdateshown','weekID','firstweekshown'))
    creatives <- trim_characters(creatives)
    creatives[,brand:=toupper(brand)]
    creatives <- creatives[brand %in% brands,]
    creatives[,dols:=as.numeric(dols)]
    
    ## remove any unknown brands
    creatives <- creatives[!like(toupper(tvcreative),"CREATIVE UNKNOWN"),]
    ## collect brands that advertised being a subset of brands
    brands.adv <- creatives[,unique(brand)]
    
    Jb = length(brands.adv)
    ## delete any brands not advertised in list - just in case..
    creatives <- creatives[brand %in% brands.adv,]
    
    ## pattern match creatives that are similar and assign cID to them
    setcreativeID(creatives)
    
    ## amount spent by a brand in a given week (this should be separated into what happened over the life of the ad,
    ## and what happened in the first week.
    ## cIDfweek and cIDlweek are introduced to refer to individual collapsed creatives (identified by cID)
    creatives[,brandadspend := sum(dols),by=c("brand","weekID")]
    csum <- creatives[,list(cIDfweek = min(cIDfweek), totalspent = sum(dols), dols.first.week = sum(ifelse(cIDfweek==firstweekshown,dols,0)), fracspent.first.week = sum(ifelse(cIDfweek==weekID,dols/brandadspend,0)), cIDlweek = max(weekID), nbrands=creatives[,uniqueN(brand)],nprogtype=uniqueN(progtype),nprograms=uniqueN(program),
    nproperty=uniqueN(property),nmedia=creatives[,uniqueN(media)], time.mins = sum(sec/60)),by=c("cID","brand")]
    
    csum[is.na(fracspent.first.week), fracspent.first.week:=0]
    ## duration of each cID creative, rounded down
    csum[,time1 := cIDlweek - cIDfweek]
    csum[,categoryname := category]
    
    ## calculate
    return(csum)
}

## converts data back from week to %d/%m/%Y
getmonth <- function(week) {
    return(month(as.Date("1979-08-27")+week*7))
}

