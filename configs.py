import os

class Config(object):

    RESIZE_DIM= 480
    INPUT_DIM = 384

    AUG_P = 0.5
    DEFORM_P = 0.9
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    DATASET_PATH = '/media/datasets/'
    DAVIS_PATH = os.path.join(DATASET_PATH, 'DAVIS/')
    VOS_PATH = os.path.join(DATASET_PATH, 'Youtube-VOS/')
    GYGO_PATH = os.path.join(DATASET_PATH, 'GyGO-Dataset')

    #DAVIS_SMALL_CATEGORIES = ['vehicle', 'animal', 'person']
    DAVIS_SMALL_CATEGORIES = ['vehicle',  'person']
    YTBVOS_SMALL_CATEGORIES = ['motorbike', 
                               'train',  'sedan',
                               'person', 'truck', 
                               'bus',]
    """
    YTBVOS_SMALL_CATEGORIES = ['motorbike', 'elephant', 
                               'train', 'cow', 'dog', 'sedan',
                               'person', 'camel', 'truck', 
                               'bird','bear', 'lion', 
                               'mouse', 'horse', 'bus',]
    """


    def display(self):
        # Display Configuration values
        print("Configurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print()
