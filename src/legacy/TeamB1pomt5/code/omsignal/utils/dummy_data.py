'''
This script generates dummy data to develop with when
we are out of priority. It needs access to the original
data, so must only be called from the server.
'''

import numpy as np 
import sys
import os

from memfile_utils import read_memfile, write_memfile

if __name__ == '__main__':
    # Load everything
    helios_data_dir = '/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/'
    dummy_data_dir = '../../data/'

    train_orig_path = os.path.join(helios_data_dir, 'MILA_TrainLabeledData.dat')
    valid_orig_path = os.path.join(helios_data_dir, 'MILA_ValidationLabeledData.dat')

    train = read_memfile(train_orig_path, shape=(160, 3754), dtype='float32')
    valid = read_memfile(valid_orig_path, shape=(160, 3754), dtype='float32')

    # Perturb the data by a constant factor...
    # This way it is not actually the original data and we can develop with it safely
    constant = 1
    train_perturb = train + constant
    valid_perturb = valid + constant

    # Write these dummy data files
    train_dummy_path = os.path.join(dummy_data_dir, 'MILA_TrainLabeledData_dummy.dat')
    valid_dummy_path = os.path.join(dummy_data_dir, 'MILA_ValidationLabeledData_dummy.dat')

    write_memfile(train_perturb, train_dummy_path)
    write_memfile(valid_perturb, valid_dummy_path)