#Constants for this particular list
#we want to inspect a specific number that is located somewhere. So we must specify 
#based on the location of the annotation of the bounding box.
#looks like the if the all the rows are below 500 and if all the columns are below 2000
#AI BASED SOLUTION
###################################################################
###################################################################
###################################################################

import subprocess 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image

DATE_COORDINATE_BOUNDS = [(-9999,500), (0,2000)]
TIME_COORDINATE_BOUNDS = [(500,800), (0,2000)]
AM_PM_COORDINATE_BOUNDS = [(800,900), (0,2000)]
TEMP_COORDINATE_BOUNDS = [(1000,1300), (0,2000)]
label_group = {'date':DATE_COORDINATE_BOUNDS, \
                'time':TIME_COORDINATE_BOUNDS,\
                'am_pm':AM_PM_COORDINATE_BOUNDS,\
                'temp':TEMP_COORDINATE_BOUNDS}

def location_tester(pred_loc, coord_bnds):
    test = all(pred_loc[:,0]>=coord_bnds[0][0]) and\
                all(pred_loc[:,0]<=coord_bnds[0][1]) and\
                all(pred_loc[:,1]>=coord_bnds[1][0]) and\
                all(pred_loc[:,0]<=coord_bnds[1][1])
    return test

def label_to_location(location, label_group):
    for k,v in label_group.items():
        if location_tester(location, v):
            return k
    return None

def date_parser(date):
    month = date[:2]
    day = date[3:5]
    year = date[-4:]
    return f'{month}/{day}/{year}'

def time_parser(time,am_pm):
    hour = int(time[:2])
    minute = time[-2:]
    if am_pm == 'pm':
        hour+=12
    return f'{hour}:{minute}:00'        

def prediction_labeler(prediction_group, label_group):
    #loops through the labels
    labels = {}
    for text in prediction_group:
        #check if this text is any of the labels
        label = label_to_location(text[1],label_group)
        if label is not None:
            if label == 'date':
                labels[label] = date_parser(text[0])
            else:
                labels[label] = text[0]
    if labels['time'] and labels['am_pm']:
        labels['time'] = time_parser(labels['time'], labels['am_pm'])
        labels.pop('am_pm',None)
    return labels

#we'll need to put a check if the labels should fail....

def label_prediction_groups(prediction_groups, label_group):
    im_list = [prediction_labeler(pred, label_group) for pred in prediction_groups]
    return im_list

###################################################################
###################################################################
###################################################################
def get_datetime(imgfile):
    search_term = 'exif:DateTimeDigitized:'
    cmd = ['identify', '-verbose', f'{imgfile}']
    out = subprocess.check_output(cmd, shell=False)
    datetime = [s for s in str(out).split('\\n') if search_term in s][0].split(' ')
    date = datetime[-2].replace(':','-')#.split(':')
    #date = f'{date[1]}/{date[2]}/{date[0]}'
    time = datetime[-1]
    datetime = f'{date} {time}'
    return datetime

def create_data_splits(df, split_ratio=0.3, seed=9999):
    train, test_ = train_test_split(df, test_size=split_ratio, random_state=seed)
    val, test = train_test_split(test_, test_size=0.5, random_state=seed)
    splits = {'train':train, 'val':val, 'test':test}
    return splits

def image_resize(im, shape=None, scale=None): 
    if shape is None:
        r,c = im.size
        shape = (int(r*scale),int(c*scale))
    resized = im.resize(shape)
    return resized

def norm_stats(df):
    m,s = np.mean(df), np.std(df)
    return m,s

def z_score(df):
    m,s = norm_stats(df)
    return (df-m)/s

def format_imagefile(filename):
    img = np.array(Image.open(filename)).astype('float32')/255.
    return img

def create_classification_data_from_dataframe(df,filecolumn='FILENAME',\
        labelcolumn='KBIN',tf_data=True):
    X = [format_imagefile(d) for d in df[filecolumn]]
    y = df[labelcolumn].astype('uint8')
    if tf_data:
        return tf.data.Dataset.from_tensor_slices((X,y))
    else:
        return X,y

def create_regression_data_from_dataframe(df,filecolumn='FILENAME',\
        labelcolumn='LOG_DISCHARGE',tf_data=True):
    X = [format_imagefile(d) for d in df[filecolumn]]
    y = np.array([df[labelcolumn].astype('float32')]).T
    if tf_data:
        return tf.data.Dataset.from_tensor_slices((X,y))
    else:
        return X,y

