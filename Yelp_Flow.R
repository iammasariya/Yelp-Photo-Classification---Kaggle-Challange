# the codes below this are in R
library(readr)
library(xgboost)
library(foreach)
library(plyr)
# read the dataset
tr <- read_csv("train.csv")
labels <- strsplit(tr$labels," ")
labels <- sapply(labels, as.integer)
labels <- lapply(labels, function(x) 0:8 %in% x)
labels <- data.frame(matrix(unlist(labels),nrow=length(labels)))
colnames(labels) <- 0:8
rownames(labels) <- tr$business_id

p2b.tr <- read_csv("train_photo_to_biz_ids.csv")
p2b.te <- read_csv("test_photo_to_biz.csv")

# read the photo-level feature data
feat.tr <- read_csv("feat03-21k.tr.csv")
feat.te <- read_csv("feat03-21k.te.csv")

# generate business-level features
gen.bfeat <- function(p2b, dat) {
  ddply(p2b, .(business_id), function(df) {
    df <- merge(df, dat)[,-(1:2)]
    ret <- llply(df, function(v) {
      c(mean=mean(v),
        sd=sd(v),
        quantile(v, c(0.6,0.8,0.9,0.95,1)))
    })
    names(ret) <- colnames(df)
    unlist(ret)
  },.progress = "text")
}

col.sd <- sapply(feat.tr[,-1], sd)
bfeat.tr <- gen.bfeat(p2b.tr, feat.tr)
bfeat.te <- gen.bfeat(p2b.te, feat.te)

# ignore features with low std deviation
sel.feat <- names(col.sd)[col.sd>0.007]
sel.col <- colnames(bfeat.tr)[sub("\\..*", "", colnames(bfeat.tr)) %in% sel.feat]
x <- as.matrix(bfeat.tr[,sel.col])
x.te <- as.matrix(bfeat.te[,sel.col])
x.te <- as.matrix(bfeat.te)

----
  #  3. Classification
  #ref: classifier chains(https://en.wikipedia.org/wiki/Classifier_chains)
  
  # train the binary classifiers
  fits <- llply(0:8, function(i) {
    x2 <- cbind(x, labels[,(0:8)<i])
    print(paste(class(labels[,(i+1)]),i,class(x2)))
    dtrain <- xgb.DMatrix(as.matrix(x2), label=labels[,(i+1)])#, missing=NA)
    
    params <- list(objective="binary:logistic", eval_metric="logloss", eta=0.01, max_depth=10, subsample=0.5, colsample_bytree=0.3)
    xgb.train(params, dtrain, nrounds=500)
  }, .progress="text")


#  4. Prediction

# calculate predictions
mat.subm <- matrix(0, nrow=nrow(x.te), ncol=9)
colnames(mat.subm) <- 0:8
for (i in 0:8) {
  target <- paste0("label", i)
  
  x2.te <- cbind(x.te, mat.subm[,(0:8)<=i])
  dtest <- xgb.DMatrix(x2.te, missing=NA)
  #dtest <- data.matrix(x2.te)
  mat.subm[,i+1] <- predict(fits[[i+1]], dtest)
}

pred <- mat.subm>0.1
colnames(pred) <- 0:8

subm <- data.frame(business_id=sort(unique(p2b.te$business_id)), labels=laply(1:nrow(pred), function(i) paste(which(pred[i,])-1, collapse=" ")))
write.csv(subm, gzfile("submmission.csv.gz"), row.names=F)