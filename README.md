# Yelp-Photo-Classification---Kaggle-Challange
Multi Label Classification for user uploaded Photos

## The description of the competition and dataset can be found here:
```
https://www.kaggle.com/c/yelp-restaurant-photo-classification 
```

## Photo-level feature extraction
- Generate the list of image files for training and testing data
```
python Generate_LST_File.py
```

- Generate mxnet data files(.rec) with following python script 
- im2rec.py is part of mxnet distribution. see tools folder in mxnet.
```
python im2rec.py --resize 224 --center_crop True tr train_photos
python im2rec.py --resize 224 --center_crop True te test_photos
```

- Generate photo-level feature using pretrained model.
- To generate feature file for train and test using any MXNet pretrained model
- I have used Inception-21k in this example
```
python Generate_Features.py
```

## Business Level feature extraction and classification are in Yelp_flow.R
