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
#' @return mcmod object being a list with the following elements: list(dimensions=dimensions,Y = Y,E = El, A = Al, X = Xl, F1 = F1l, F2 = F2)
#' @examples
#' mcmod() - Runs default with "dpp"
#' mcmod(data.name = "dpp", brands.to_keep = c('HUGGIES','PAMPERS','LUVS','PL'), covv = c("avprc"), covnv = c("fracfnp","fracdnp","fracdist","numproducts"), T = 226,  N = 42, fweek = 1200, aggregated = FALSE)
mcmodf <- function(data.name = "dpp", brands.to_keep = c('HUGGIES','PAMPERS','LUVS','PL'), covv = c("avprc"), covnv = c("fracfnp","fracdnp","fracdist","numproducts"), T = 226,  N = 42, fweek = 1200, aggregated = FALSE) {

    J = length(brands.to_keep)
    P <- J * length(covv)          # total number of time varying covariates at city level
    # non time varying componentns
    K <- J*length(covnv)

    category <- data.name

    ###### read in data file required and make any transforms needed
    if(aggregated) cf <- paste(category,'agg',sep="") else cf <- category
    dfname <- sprintf('nobuild/data-raw/%siritns.txt',cf)

    ## Trim characters, just in case
    df1 <- fread(dfname)
    df1 <- trim_characters(df1)
    
    market_list <- unique(df1$market_name)


        if(any(brands.to_keep == "OTHER"))
    df1 <- rbind(df1, df1[!brand %in% brands.to_keep[-which(brands.to_keep == "OTHER")] ,list(brand="OTHER",volume=sum(volume),units=sum(units),dollars=sum(dollars),lavgprc = mean(lavgprc), sumfeature=mean(sumfeature),sumdisplay=mean(sumdisplay),sumfnp=mean(sumfnp),sumdnp=mean(sumdnp),numproducts=mean(numproducts),numoutlets=mean(numoutlets),totalnumstores=min(totalnumstores),est_acv=mean(est_acv),tot_acv=min(tot_acv),fvol=sum(fvol),dvol=sum(dvol),pvol=sum(pvol),spotadsecs=sum(spotadsecs,na.rm=TRUE),spotaddols=sum(spotaddols,na.rm=TRUE),spotadunits=sum(spotadunits,na.rm=TRUE),natsecs=sum(natsecs,na.rm=TRUE),natdols=sum(natdols,na.rm=TRUE),natunits=sum(natunits,na.rm=TRUE),networkadsecs=sum(networkadsecs,na.rm=TRUE),networkaddols=sum(networkaddols,na.rm=TRUE),networkadunits=sum(networkadunits,na.rm=TRUE),syndicationadsecs=sum(syndicationadsecs,na.rm=TRUE),syndicationaddols=sum(syndicationaddols,na.rm=TRUE),syndicationadunits=sum(syndicationadunits,na.rm=TRUE),cableadsecs=sum(cableadsecs,na.rm=TRUE),cableaddols=sum(cableaddols,na.rm=TRUE),cableadunits=sum(cableadunits,na.rm=TRUE),SLNadsecs=sum(SLNadsecs,na.rm=TRUE),SLNaddols=sum(SLNaddols,na.rm=TRUE), SLNadunits=sum(SLNadunits,na.rm=TRUE),avprc=mean(avprc),fracdnp=mean(fracdnp),fracfnp=mean(fracfnp),fracdist=mean(fracdist)),by=c("market_name","week")])


    # keep only brands focused on here, and subset of weeks and markets
    df1 <- df1[brand %in% brands.to_keep & week < fweek + T & week >= fweek & market_name %in% market_list[1:N]]

    # only do the rest if balanced
    if(any(dcast.data.table(df1, week ~ market_name, value.var="volume", subset = .(week < fweek+T & week >= fweek & market_name %in% market_list[1:N]), fun = length) < J)) {
        warning("Warning: Cannot do analysis on brands with unbalanced data: reduce brands in brands.to_keep, or choose a subset of markets. Step unable to complete for analysis, but ok for summary data (use df1)")
    } else {
        # create new or transformed variables here, if required
        #df1[,sumdnp := log(1+df1$sumdnp)]
        #df1$sumdnp<-log(1+df1$sumdnp)
        #df1$sumfnp<-log(1+df1$sumfnp)

        # create outcome variable (Y)
        Y <- structure(rep(0,T*N*J), dim=c(T,N,J))
        for(j in 1:J) {
            .a <- dcast.data.table(df1, week ~ market_name, value.var = "volume", subset = .(week >= fweek & brand %in% brands.to_keep[j]), fill=0, fun = min)
            .a[,week := NULL]
            Y[,,j] <- simplify2array(.a)
        }
        rm(.a)


        # create advertising variable (XAdv), national only here (using mean, therefore)
        df1[is.na(natdols),natdols:=0]
        XAdv <- dcast.data.table(df1, week ~ brand, value.var = "natdols", subset = .(week >= fweek), fill = 0, fun = mean, na.rm = TRUE)
        XAdv <- XAdv[,brands.to_keep,with=FALSE]			# reorder columns to correspond to list

        # create covariates from list, X being not time varying, and X2 being time varyin

        X1 <- structure(rep(0,T*N*J*length(covnv)), dim=c(T,N,J*length(covnv)))
        for(mkt in 1:N) {
            for(j in 1:J) {
                .a <- dcast.data.table(df1, week ~ market_name, value.var = covnv, subset = .(week >= fweek & market_name == market_list[mkt] & brand == brands.to_keep[j]), fun = mean, na.rm = TRUE)
                .a[,week := NULL]
                if(j ==1) .xcov <- simplify2array(.a) else .xcov <- cbind(.xcov, simplify2array(.a))
            }
            X1[,mkt,] <- .xcov
        }
        rm(.a,.xcov)

        X2 <- structure(rep(0,T*N*J*length(covv)), dim=c(T,N,J*length(covv)))
        for(mkt in 1:N) {
            for(j in 1:J) {
                .a <- dcast.data.table(df1, week ~ market_name, value.var = covv, subset = .(week >= fweek & market_name == market_list[mkt] & brand == brands.to_keep[j]), fun = mean, na.rm = TRUE)
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
    .brands_advertised <- as.numeric(which(colSums(A)>0))
    if(any(diff(which(colSums(A)>0))>1)) stop("Reorder columns for advertised brands, unless you make sure all 1:Jb brands advertise, the estimation will likely be incorrect.")
    Jb <- length(.brands_advertised)
    brands.adv <- brands.to_keep[.brands_advertised]
    A <- A[,.brands_advertised]

    ## for this subset, which brands launched new creatives in that time frame?
    # call creatives function for Jb brands
    ownadnnc <- getcreatives(category, brands.adv, fweek, T, make.binary=FALSE)
    #load(paste(codepath,"/",category,"creatives.RData",sep=""))
    s.nnc <- as.data.frame(ownadnnc[weekID >= fweek & weekID < fweek+T,])
    s.nnc$weekID <- NULL

    .brands_changed <- which(colSums(s.nnc)>0)
    brands_nnc <- brands.adv[.brands_changed]

    ## Data for creatives - note if any adv = 0 then columns removed
    ## 2/2/2015 - now E is same dimension as ads (Jb) but JbE measures the number of brands with observed changes so JbE could be less than Jb
    ##    E <- s.nnc[1:T,.brands_changed]
    E <- s.nnc[1:T,]
    JbE <- length(brands_nnc)
    JbEv <- match(brands_nnc, brands.adv)			# pointer to brands.adv, which ones changed


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
    Al <- El <- Xl <- F1l <- Yl <- list()
    for (t in 1:T) {
   ##     El[[t]] <- E[t, ]
        El[[t]] <- as.numeric(E[t,])
   ##     names(El[[t]]) <- colnames(E)
        Al[[t]] <- A[t, ]
        Xl[[t]] <- X1[t,,]
        F1l[[t]] <- t(as(F1[t,,], "dgCMatrix"))
	Yl[[t]] <- Y[t,,]
    }

    dimensions <- list(N = N, T= T, J=J, Jb = Jb, JbE = JbE, K = ncol(Xl[[1]]), P = P)

    mcmod <- list(dimensions=dimensions,Y = Yl,E = El, A = Al, X = Xl, F1 = F1l, F2 = F2)
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
##' @param make.binary Flag (T or F) for whether E should be dichotomous
##' @return T x Jb matrix with integer corresponding to number of new creatives added that week
##' @examples
##' getcreatives(category, brands.adv, Jb, fweek, T, make.binary=F) - called from within function mcmod()
getcreatives <- function(category, brands.adv, fweek, T, max.distance=0.2, make.binary=FALSE) {

    Jb = length(brands.adv)
    cf <- paste0("nobuild/data-raw/",category,"creatives.txt")
    ## convert brand names

    creatives <- fread(cf,col.names= c('brand','program','progtype','tvcreative','property','media','avg30','avg30d','dols','sec','dtime','firstdateshown','weekID','fweek'))

    ## Trim fields here just in case
    creatives <- trim_characters(creatives)
    
    for(j in 1:Jb) creatives[agrep(brands.adv[j],brand,ignore.case=T), brand:= brands.adv[j]]
    creatives <- creatives[brand %in% brands.adv,]			# delete any brands not advertised in list

    creatives[,ENEWS:=0]
    #### now generic code

    ## group creatives together and create a new creativeID (cID) for each
    ## Note that "other" creatives are assigned an ID of 5000 and all grouped together
    
    #### now generic code
    ## collapse
    
    creatives[,cID := (Jb+1)*1000]
    for(j in 1:Jb){
        li<-list()
        .c <- creatives[brand==brands.adv[j],unique(tvcreative)]
        li[[1]]<-clist<-agrep(.c[1],.c,max.distance=max.distance)
        mc<-1
        for(m in 2:length(.c)) if(!any(clist==m)) {
            mc<-mc+1
            li[[mc]]<-agrep(.c[m],.c,max.distance=max.distance)	# max.distance is measuring how difft two creative descriptors are
            clist<-c(clist,li[[mc]])
        }
        for(m in 1:length(li)) creatives[tvcreative %in% .c[li[[m]]] & toupper(brand)==brands.adv[j],cID:= j * 1000 + m]
    }
    
    #
    creatives[,cIDfweek:=min(fweek),by=cID]
    
    # count new ads (just use different labels, perhaps can improve on this with distance)
    creatives[weekID==cIDfweek, nnc := uniqueN(cID),by=c("weekID","brand")]
    creatives[is.na(nnc), nnc := 0]
    
    if(make.binary) creatives[nnc>0,nnc:=1]

    # collapse to get total new creatives
    .nnc <- creatives[,max(nnc),by=c("weekID","brand")]

    ownadnnc <- merge(creatives[,.N,by=weekID], creatives[brand == brands.adv[1],max(nnc),by=c("weekID")],by="weekID",all.x=TRUE,all.y=F)
    setnames(ownadnnc,"V1",brands.adv[1])
    for(b in 2:Jb) {
        ownadnnc <- merge(ownadnnc, creatives[brand == brands.adv[b],max(nnc),by=c("weekID")],by="weekID",all.x=TRUE,all.y=F)
        setnames(ownadnnc,"V1", brands.adv[b])
    }
    ownadnnc <- ownadnnc[,N := NULL]

    allweeks <- data.table(weekID=min(ownadnnc$weekID):max(ownadnnc$weekID))
    setkey(allweeks,weekID)
    setkey(ownadnnc,weekID)

    ownadnnc <- merge(allweeks,ownadnnc,all.x=TRUE)
    ownadnnc[is.na(ownadnnc)] <- 0
    return(ownadnnc)
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
