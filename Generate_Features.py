# mxnet prediction using pretrained model

import mxnet as mx
import numpy as np
import pandas as pd
import logging

import glob
import re
import os
import time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

BATCHSIZE = 100

# for train data
prefix = 'tr'
NUMBATCH = 2349
outfile = 'feat03-21k.tr.csv'

# for test data
#prefix = 'te'
#NUMBATCH = 2372
#outfile = 'feat03-21k.te.csv'

model_dir = 'inception-21k/'
model_prefix = os.path.join(model_dir, 'Inception')
num_round = 9


start_time = time.time()

model = mx.model.FeedForward.load(model_prefix, num_round, gtx=num_cpu(),numpy_batch_size=1)


dataiter = mx.io.ImageRecordIter(
    path_imgrec=prefix + '.rec',
    data_shape=(3,224,224),
    batch_size=BATCHSIZE,
    mean_r=117.0,
    mean_g=117.0,
    mean_b=117.0,
    preprocess_threads=6,
    prefetch_buffer=1)
# to avoid large output file containing all the irrelevant predictions,
# i have sampled photo ids and saved column numbers of 2000 columns
# which have larger std dev in 'col.sel.csv'
col_sel = pd.read_csv('col.sel.csv')


first = True
for i in range(NUMBATCH):
  batch = dataiter.next()
  idx = batch.index
  pad = batch.pad
  data = batch.data[0]

  prob = model.predict(data)
  prob = prob[:, col_sel.col-1]

  if pad > 0:
    N = len(idx) - pad
    idx = idx[:N]
    prob = prob[:N]

  df = pd.concat([pd.DataFrame({'photo_id':idx}), pd.DataFrame(prob, columns=['feat'+str(j+1) for j in range(prob.shape[1])])], axis=1)

  header = first
  mode = 'w' if first else 'a'
  df.to_csv(outfile % SEQ, index=False, header=header, mode=mode)
  first = False

  if (i+1) % 100 == 0:
    print('%d processed: %ds' % ((i+1), time.time() - start_time))

print('done; elapsed = %ds' % (time.time() - start_time))

