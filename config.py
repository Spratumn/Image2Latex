class Config:
    # system
    DATA_DIR = 'data'
    LOG_DIR = 'logs'
    CSV_DIR = 'data_list'
    WEIGHTS_DIR = 'logs/weights'
    OUTPUT_DIR = 'output'
    GPU = [0]
    DEVICE = 'cuda'
    NUM_WORKERS = 4  # 'dataloader threads. 0 for single-thread.'

    # model
    DOWN_RATIO = 4  # output stride
    # encoder
    INPUT_SIZE_ENCODER = 256
    HIDDEN_SIZE_ENCODER = 512
    # decoder
    INPUT_SIZE_DECODER = 128
    HIDDEN_SIZE_DECODER = 512
    OUTPUT_SIZE_DECODER = 10
    SEQUENCE_NUM = 5

    # dataset
    # input


    # train
    LR = 1.0e-5  # learning rate.
    EPOCHS = 10  # total training epochs.
    BATCH_SIZE = 2

    # loss
    HM_WEIGHT = 1  # loss weight for key point heat maps
    OFF_WEIGHT = 1  # loss weight for key point local offsets
    WH_WEIGHT = 2  # loss weight for bounding box size
    EPS = 1e-8




