import pandas as pd

# read data
p2b_tr = pd.read_csv('dataset/train_photo_to_biz_ids.csv')
p2b_te = pd.read_csv('dataset/test_photo_to_biz.csv')

rows = []
for pid in set(p2b_tr.photo_id):
  rowdict = { 'photo_id':pid, 'class':0, 'filename':'%d.jpg' % pid }
  rows.append(rowdict)
df_train = pd.DataFrame(rows)
df_train.to_csv('tr.lst', columns=['photo_id', 'class', 'filename'], sep='\t', header=False, index=False)

rows = []
for pid in set(p2b_te.photo_id):
  rowdict = { 'photo_id':pid, 'class':0, 'filename':'%d.jpg' % pid }
  rows.append(rowdict)
df_test = pd.DataFrame(rows)
df_test.to_csv('te.lst', columns=['photo_id', 'class', 'filename'], sep='\t', header=False, index=False)


