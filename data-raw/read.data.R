## read.data.R - creates IRI and TNS data
trim <- function (x) gsub("^\\s+|\\s+$", "", x)
cname <- "COMPOSITEBRAND"

datapath <- "~/Dropbox/Research/Data/IRI5"
thispath <- "./data-raw/"

###############################################################################################################
# Read from classification file
# data.name		IRIfile		TNSfile		
cDT <- fread(paste0(thispath, "classification.txt"))
cDT <- cDT[bmd.data.name == data.name,]

###############################################################################################################
## IRI Data

categories <- list.files(path=paste0(datapath, "/Year6/External"))
if(!any(match(cDT[,unique(iricatname)],categories))) stop("No categories matching in iri for: ", data.name,"\n")

## augment the following category if there are more than one
#iridata <- NULL		
#for(category in cDT[,unique(iricatname)]) {
	category <- cDT[,unique(iricatname)]
	## available superset of IRI data
	years <- 1:6

	IRITNSBrand <- data.frame(CATEGORY=character(),TNSBRAND = character(), IRIBRAND = character())

	## read in UPC data
	require(gdata)
	fn <- paste0(datapath, "/parsed stub files/prod_", category,".xls")
	.udata <- data.table(read.xls(fn))
	colnames(.udata)[13]<-"STUBSPEC"

	## select subset of UPC data here
	udata <- .udata[,list(PARENT=L3,VENDOR=L4,IRIBRAND=L5,VOL_EQ,SY,GE,VEND,ITEM)]

	## read in IRI main data

	yl <- list()
	yl[[1]] <- c(1114,1165)
	yl[[2]] <- c(1166,1217)
	yl[[3]] <- c(1218,1269)
	yl[[4]] <- c(1270,1321)
	yl[[5]] <- c(1322,1373)
	yl[[6]] <- c(1374,1426)

	mdata <- NULL
	for(year in years) {
	
		# store level data for location (market name) and for est_acv
		dn <- paste0(datapath,"/Year",year,"/External/",category,"/Delivery_Stores")
		fn1 <- paste0(datapath,"/Year",year,"/External/",category,"/",category,"_groc_",yl[[year]][1],"_",yl[[year]][2])	
		fn2 <- paste0(datapath,"/Year",year,"/External/",category,"/",category,"_drug_",yl[[year]][1],"_",yl[[year]][2])	
		m1 <- fread(fn1)
		m2 <- fread(fn2)

		## read in delivery store data	
		.ds <- data.table(read.fwf(dn,widths=c(7,-1,2,9,-1,25,4,5,-1,8),skip=1,col.names = c("IRI_KEY","OU","EST_ACV","MarketName","StoreOpen","StoreClosed","MskdName")),key="IRI_KEY")
		.ds[,OU := NULL]
		.ds[,MskdName := NULL]
		.ds <- unique(.ds)				# in some case there are duplicate rows.
		.ds[,MKT_EST_ACV := sum(EST_ACV), by = MarketName]
	
		## merge the two for each year
		.mdata <- rbind(m1,m2)
		.mdata[,catname := category,]
		rm(m1,m2)
		setkey(.mdata,IRI_KEY)

		mdata <- rbind(mdata,.ds[.mdata])
		}

	## combine UPC and IRI data
	setkey(mdata,SY,GE,VEND,ITEM)
	setkey(udata,SY,GE,VEND,ITEM)

	a <- udata[mdata]
	setnames(a, toupper(colnames(a)))
	a[,MARKETNAME := trim(MARKETNAME)]

    ## use the below to see which brands are around for all weeks and all
    ##	a[,uniqueN(WEEK),by=c("MARKETNAME","IRIBRAND")][V1==313,uniqueN(MARKETNAME),by=IRIBRAND][V1==50]

	.brands.full <- a[,unique(IRIBRAND)]			## full set of brands

	## code up UPC for counting
	a[,UPC := VEND*100000+ITEM]
	a[,numupc := uniqueN(UPC), by = c("MARKETNAME","WEEK","IRIBRAND")]		# per variant
	a[,totalnumstores := uniqueN(IRI_KEY),by= c("MARKETNAME","WEEK")]		# total number of stores in market for that week
	a <- a[UNITS>0,]
	a[,UNITS := as.numeric(UNITS)]
	a[,price := DOLLARS/(UNITS*VOL_EQ)]
	a[,FD := ifelse(F=="NONE",0,1)]			# feature dummy
	a[,DD := ifelse(D==0,0,1)]				# display dummy
	a[,PRD := ifelse(PR==0,0,1)]			# price promo dummy

	dmake <- function(.dtf) {
		## make denominator for distribution weighted by ACV
		.t <- .dtf[,min(EST_ACV),by=c("IRI_KEY","MARKETNAME","WEEK")][,list(SE=sum(V1)),by=c("MARKETNAME","WEEK")]
		setkey(.t,MARKETNAME,WEEK)
		setkey(.dtf,MARKETNAME,WEEK)
		.tt <- .t[.dtf]
		.tt[,list(numupc = uniqueN(UPC), dollars = sum(DOLLARS), volume = sum(UNITS*VOL_EQ), lavgprc = mean(log(price)), feature = weighted.mean(FD,EST_ACV), display = weighted.mean(DD,EST_ACV), priceoff = weighted.mean(PRD, EST_ACV), numoutlets = uniqueN(IRI_KEY), numproducts = log(uniqueN(UPC)),
	totalnumstores = max(totalnumstores),est_acv = mean(EST_ACV), mkt_est_acv = min(MKT_EST_ACV), fracwdist = min(SE/MKT_EST_ACV)), by = c("MARKETNAME","WEEK","BRAND")]
	}

	fdata <- sub.brands <- NULL
	for(b in brandlist) {
		.bs <- as.character(.brands.full[agrep(b,.brands.full)])
		sub.brands <- c(sub.brands,.bs)
		.dtf <- a[IRIBRAND %in% .bs,]		# does a pattern match on provided brand names
		.dtf[,BRAND := b]
	
		fdata <- rbind(fdata, dmake(.dtf))	
		}	
	## now add in composite
	.dtf <- a[!IRIBRAND %in% sub.brands,]
	.dtf[,BRAND := cname]

	.iridata <- rbind(fdata, dmake(.dtf))	

	setnames(.iridata, tolower(colnames(.iridata)))			# lower case column names

	## additional variables
	.iridata[,avgprc := dollars/volume]

	setnames(.iridata, "marketname", "market_name")

	## clean up
	rm(sub.brands,.bs,.dtf)

	## restriction to markets we had earlier
    if(select.markets.from.file) {
        source(paste0(thispath,'market_list.R'))
        .iridata <- .iridata[market_name %in% market_list]
    }
	setkey(.iridata,market_name,week,brand)
	
	iridata <- .iridata	
##	iridata <- rbind(iridata,.iridata)
##}

## save to the right location
saveRDS(iridata, file=sprintf("~/Documents/ads_class/nobuild/data-raw/%siri.RDS",data.name))

###############################################################################################################
## Read from TNS Data

header <- colnames(fread(paste0(datapath, "/TNSData/header.txt")))

## TNSdata <- fread(paste0(datapath, "/TNSData/",category,"/creatives.txt"), col.names=toupper(header))
TNSdata <- fread(paste0(datapath, "/TNSData/TNSdata.txt"), col.names=toupper(header))
TNSdata[,TNSBRAND := BRAND]
TNSdata[,BRAND:=NULL]

## clean up
TNSdata <- TNSdata[!TNSBRAND %in% "Brand Not Identified",]

## this next step allows for multiple categories to be combined
## note that any brands that are common across the subcategories will also be combined here
.tDT <- TNSdata[SUBCATEGORY %in% cDT[,tnssubcat] & MICROCATEGORY %in% cDT[,tnsmiccat],]
rm(TNSdata)

.tDT[,DATE := ymd_hms(DTIME)]
.tDT[,WEEK := as.integer((DATE-ymd_hms("1979-08-27 00:00:00.000"))/7)]

brands.advertised <- .tDT[,unique(TNSBRAND)]

.tDT[,FIRSTDATESHOWN := min(DATE),by = c("TNSBRAND","TVCREATIVE")]
.tDT[,FIRSTWEEKSHOWN := as.integer((FIRSTDATESHOWN - ymd_hms("1979-08-27 00:00:00.000"))/7)]
TNS <- .tDT[,list(BRAND=TNSBRAND,PROGRAM,PROGTYPE,TVCREATIVE,PROPERTY,MEDIA,AVG30,AVG30D,DOLS,SEC,DTIME,FIRSTDATESHOWN,WEEKID=WEEK,FIRSTWEEKSHOWN)]
colnames(TNS) <- tolower(colnames(TNS))
setnames(TNS,"weekid","weekID")
saveRDS(TNS, file=sprintf("~/Documents/ads_class/nobuild/data-raw/%stns.RDS",data.name))
	