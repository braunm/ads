 mcmod <- mcmod(data.name = "dpp", brands.to_keep = c('HUGGIES','PAMPERS','LUVS','PL'), covv = c("avprc"), covnv = c("fracfnp","fracdnp","fracdist","numproducts"), T = 226,  N = 42, fweek = 1200, aggregated = FALSE)
 
 save(mcmod, file="data/mcmoddpp.RData")
 