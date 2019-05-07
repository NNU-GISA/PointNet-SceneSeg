import os
import numpy as np
import sys
import h5py
import shutil
import vkitti_util

# constant
NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
data_dir = os.path.join(ROOT_DIR, 'data')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
vkitti_data_dir_origin = os.path.join(data_dir, 'vkitti3d_dataset_v1.0')
vkitti_data_dir = os.path.join(data_dir, 'vkitti3d_dataset_rename')
if not os.path.exists(vkitti_data_dir):
    os.mkdir(vkitti_data_dir)
output_dir = os.path.join(data_dir, 'vkitti_hdf5_test_1')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
# filelist txt
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')
# h5 file txt
hdf5_filelist = os.path.join(output_dir, 'all_files.txt')
fout_hdf5 = open(hdf5_filelist, 'w')

########################################## BATCH WRITE TO HDF5 ##########################################
batch_data_dim = [H5_BATCH_SIZE] + data_dim # state: [1000, 4096, 9]
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

## function hdf5 in batch
def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        # stack data and label in order
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        # write all_files.txt 1
        fout_hdf5.write(output_filename_prefix + '_' + str(h5_index)+ '.h5' + '\n')
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        # write all_files.txt 2
        fout_hdf5.write(output_filename_prefix + '_' + str(h5_index) + '.h5' + '\n')
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return

## Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

############################################# RENAME NPY ################################################
def move_file(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile,dstfile)
        print ("move %s -> %s"%( srcfile,dstfile))

def copy_file(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(srcfile,dstfile)
        print ("copy %s -> %s"%( srcfile,dstfile))

for root, dirs, files in os.walk(vkitti_data_dir_origin):
    #print('root', root)
    #print('dirs', dirs)
    #print('files', files)
    for srcfile in files:
        print(srcfile, root[-1:])
        copy_file(os.path.join(root,srcfile), os.path.join(vkitti_data_dir,'Area_{}_{}'.format(root[-1:], srcfile)))

filelist = []
for root, dirs, files in os.walk(vkitti_data_dir):
    for file in files:
        filelist.append(os.path.join(root,file))
print(filelist)

############################################## HDF5 MAIN ################################################
sample_cnt = 0

# data sample
for i, data_label_filename in enumerate(filelist):
    print(data_label_filename)

    data, label = vkitti_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=6.0, stride=3,
                                                 random_sample=False, sample_num=None)
    print('data shape: {0}, label shape: {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    # save hdf5 file
    insert_batch(data, label, i == len(filelist)-1)

# txt file close
fout_room.close()
fout_hdf5.close()
print("Total samples: {0}".format(sample_cnt))
