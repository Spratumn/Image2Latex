class Config:
    # system
    DATA_DIR = 'data'
    FORMULAS_DIR = 'C:/Users/QIU/Desktop/dataset/image2latex100k/im2latex_formulas.lst'
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
    HIDDEN_SIZE_DECODER = 512
    MAX_FORMULA_LENGTH = 512

    # dataset
    CROP_SIZE = []
    TOKEN_COUNT = 1157

    # train
    LR = 1.0e-5  # learning rate.
    EPOCHS = 10  # total training epochs.
    BATCH_SIZE = 1

    # loss
    HM_WEIGHT = 1  # loss weight for key point heat maps
    OFF_WEIGHT = 1  # loss weight for key point local offsets
    WH_WEIGHT = 2  # loss weight for bounding box size
    EPS = 1e-8




