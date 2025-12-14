import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MovingMNIST or UCF101
    DATASET_NAME = "UCF101"
    IMG_SIZE = 64
    CHANNELS = 1

    # We take t frames input -> Predict next k frames
    INPUT_FRAMES = 5
    PRED_FRAMES = 5
    TOTAL_FRAMES = INPUT_FRAMES + PRED_FRAMES


    LATENT_DIM = 64

    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 500

    # hyperparameters from original paper
    LAMBDA_IDENTITY = 1.0  # reconstruction
    NU_BACKWARD = 0.5  # backward prediction
    ETA_CONSISTENCY = 0.01  # matrix consistency
