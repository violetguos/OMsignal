"""a files to store all constants, such as fixed dimensions, path names"""
import os

# path to REAL data
REAL_OMSIGNAL_PATH = "/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/"
TRAIN_LABELED_DATASET_PATH = os.path.join(
    REAL_OMSIGNAL_PATH, "MILA_TrainLabeledData.dat"
)
VALID_LABELED_DATASET_PATH = os.path.join(
    REAL_OMSIGNAL_PATH, "MILA_ValidationLabeledData.dat"
)
UNLABELED_DATASET_PATH = os.path.join(REAL_OMSIGNAL_PATH, "MILA_UnlabeledData.dat")

# dimensions, indices
LABELED_SHAPE = (160, 3754)
PR_MEAN_IDX = 3750
RT_MEAN_IDX = 3751
RR_STDDEV_IDX = 3752
ID_IDX = 3753

UNLABELED_SHAPE = (657233, 3750)
SHAPE_OF_ONE_DATA_POINT = (1, 3750)
SIZE_OF_DATA_POINT_BYTES = 3750 * 4 # data dim times size of a float32

# legacy code path
TA_LEGACY_CODE = (
    os.path.dirname(os.path.abspath(__file__)) + "/../legacy/TABaseline/code"
)
TA_LEGACY_MODEL = (
    os.path.dirname(os.path.abspath(__file__)) + "/../legacy/TABaseline/model/baseline_final.pt"
)

T5_LEGACY_CODE = (
    os.path.dirname(os.path.abspath(__file__)) + "/../legacy/TeamB1pomt5/code"
)
T5_LEGACY_MODEL = (
    os.path.dirname(os.path.abspath(__file__)) + "/../legacy/TeamB1pomt5/model"
)

# fake data path, credits to team 5
T5_FAKE_TRAIN_LABELED_DATA = (
    os.path.dirname(os.path.abspath(__file__))
    + "/../legacy/TeamB1pomt5/code/data/MILA_TrainLabeledData_dummy.dat"
)
T5_FAKE_VALID_LABELED_DATA = (
    os.path.dirname(os.path.abspath(__file__))
    + "/../legacy/TeamB1pomt5/code/data/MILA_ValidationLabeledData_dummy.dat"
)

TARGETS = 'pr_mean, rt_mean, rr_stdev, userid'
