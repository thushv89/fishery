__author__ = 'Thushan Ganegedara'

from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os
from scipy import ndimage
from math import floor,ceil
from skimage.measure import block_reduce
from scipy.misc import imsave

image_width,image_height,num_channels = 1280,720,3

def load_fishery_data(resize=True,test_image=False,test_pickle=False):

    augment_data = False

    fishery_train_dir = 'train'
    fishery_test_dir = 'test_stg1'
    cropped_width,cropped_height = 640,640 #min width 1192, min height 670
    labels = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']

    valid_size = 2000
    if not os.path.exists(fishery_train_dir):
        raise FileNotFoundError


    min_width,min_height = 5000,5000
    find_min = False

    for l_i,lbl in enumerate(labels):
        if test_pickle and l_i==0:
            continue

        print("Processing folder %s ..."%lbl)
        img_dataset = []
        img_labels = []
        pixel_depth = -1
        for file in os.listdir(fishery_train_dir+os.sep+lbl):
            if file.endswith(".jpg"):
                image_data = ndimage.imread(fishery_train_dir+os.sep+lbl+os.sep+file).astype(float)
                if not find_min:
                    # there's no specific size it changes
                    #assert image_data.shape == (image_height,image_width,num_channels)
                    if pixel_depth == -1:
                        pixel_depth = 255 if np.max(image_data)>128 else 1
                    # crop from right
                    cut_from_right,cut_from_left = floor(float(image_data.shape[1]-cropped_width)/2.0),ceil(float(image_data.shape[1]-cropped_width)/2.0)

                    image_data = np.delete(image_data,np.r_[0:cut_from_right],axis=1)
                    image_data = np.delete(image_data,np.r_[image_data.shape[1]-cut_from_left: image_data.shape[1]],axis=1)

                    if image_height != cropped_height:
                        cut_from_top,cut_from_bottom = floor(float(image_data.shape[0]-cropped_height)/2.0),ceil(ceil(image_data.shape[0]-cropped_height)/2.0)
                        image_data = np.delete(image_data,np.r_[0:cut_from_top],axis=0)
                        image_data = np.delete(image_data,np.r_[image_data.shape[0]-cut_from_bottom:image_data.shape[0]],axis=0)

                    assert image_data.shape == (cropped_height,cropped_width,num_channels)
                    image_data = (image_data - (pixel_depth/2.0))/(pixel_depth/2.0)

                    if resize:
                        image_data = block_reduce(image_data,(2,2,1))

                    img_dataset.append(image_data)
                    img_labels.append(l_i)

                    if test_image:
                        imsave('test_'+lbl+'.png', image_data)
                        break

                else:
                    if image_data.shape[1]<min_width:
                        min_width = image_data.shape[1]
                    if image_data.shape[0]<min_height:
                        min_height = image_data.shape[0]
        try:
            with open('fishery_'+lbl+'.pickle', 'wb') as f:
                img_dataset = np.asarray(img_dataset)
                img_labels = np.asarray(img_labels)
                print("Dataset shape %s"%str(img_dataset.shape))
                print("Labels shape %s"%str(img_labels.shape))
                pickle.dump({'dataset':image_data,'labels':img_labels}, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save fishery_data (%s): %s' %(lbl,str(e)))

        if test_pickle:
            break

    print('Composing test data ...')

    del img_dataset
    del img_labels
    test_dataset = []
    test_labels = []
    if not test_image and not test_pickle:
        for file in os.listdir(fishery_test_dir):
            if file.endswith(".jpg"):
                image_data = ndimage.imread(file).astype(float)
                assert image_data.shape == (image_width,image_height,num_channels)
                # crop from right
                image_data = np.delete(image_data,np.r_[0:(image_width-cropped_width)//2],axis=1)
                # crop from left
                image_data = np.delete(image_data,np.r_[-(image_width-cropped_width)//2:image_data.shape[1]],axis=1)

                image_data = (image_data - (pixel_depth/2.0))/(pixel_depth/2.0)
                if image_height != cropped_height:
                    raise NotImplementedError

                test_dataset.append(image_data)
                test_labels.append(l_i)

        test_dataset = np.asarray(test_dataset)
        test_labels = np.asarray(test_labels)

        print("Test dataset shape %s"%test_dataset.shape)
        print("Test labels shape %s"%test_labels.shape)

        print('Successfully whitened data ...\n')
        try:
            with open('fishery-test-stg1.pickle', 'wb') as f:
                pickle.dump({'dataset':test_dataset,'labels':test_labels}, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save fishery_data:', e)


def reformat_data_fishery():

    image_size = 32
    num_labels = 10
    num_channels = 3 # rgb

    print("Reformatting data ...")
    cifar10_file = '..'+os.sep+'data'+os.sep+'cifar_train.pickle'
    with open(cifar10_file,'rb') as f:
        save = pickle.load(f)
        train_dataset, train_labels = save['train_dataset'],save['train_labels']
        valid_dataset, valid_labels = save['valid_dataset'],save['valid_labels']
        test_dataset, test_labels = save['test_dataset'],save['test_labels']

        train_dataset = train_dataset.reshape((-1,image_size,image_size,num_channels)).astype(np.float32)
        valid_dataset = valid_dataset.reshape((-1,image_size,image_size,num_channels)).astype(np.float32)
        test_dataset = test_dataset.reshape((-1,image_size,image_size,num_channels)).astype(np.float32)

        print('\tFinal shape (train):%s',train_dataset.shape)
        print('\tFinal shape (valid):%s',valid_dataset.shape)
        print('\tFinal shape (test):%s',test_dataset.shape)

        train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
        valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
        test_labels = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)

        print('\tFinal shape (train) labels:%s',train_labels.shape)
        print('\tFinal shape (valid) labels:%s',valid_labels.shape)
        print('\tFinal shape (test) labels:%s',test_labels.shape)

    return (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)
