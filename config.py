'''
Configuration options for MetaRegGNN pipeline.
'''


class Config:
    # SYSTEM OPTIONS
    DATA_FOLDER = './simulated_data/'  # path to the folder data will be written to and read from
    RESULT_FOLDER = './'  # path to the folder data will be written to and read from

    # SIMULATED DATA OPTIONS
    CONNECTOME_MEAN = 0.0  # mean of the distribution from which connectomes will be sampled
    CONNECTOME_STD = 1.0  # std of the distribution from which connectomes will be sampled
    SCORE_MEAN = 90.0  # mean of the distribution from which scores will be sampled
    SCORE_STD = 10.0  # std of the distribution from which scores will be sampled
    N_SUBJECTS = 30  # number of subjects in the simulated data
    ROI = 116  # number of regions of interest in brain graph
    SPD = True  # whether or not to make generated matrices symmetric positive definite

    # EVALUATION OPTIONS
    K_FOLDS = 3  # number of cross validation folds

    # MetaRegGNN OPTIONS
    class MetaRegGNN:
        NUM_EPOCH = 300  # number of epochs the process will be run for
        LR = 1e-3  # learning rate
        WD = 5e-4  # weight decay
        DROPOUT = 0.2  # dropout rate
        #meta-learning options
        GAMMA = 0.0000001 
        ETA = 1e-3

    # RANDOMIZATION OPTIONS
    DATA_SEED = 1  # random seed for data creation
    MODEL_SEED = 1  # random seed for models
    SHUFFLE = True  # whether to shuffle or not
