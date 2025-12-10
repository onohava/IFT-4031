import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    DATA_PATH = "./data"
    IMG_SIZE = 64
    CHANNELS = 1

    # We take 5 frames input -> Predict next 5 frames
    INPUT_FRAMES = 1
    PRED_FRAMES = 5
    TOTAL_FRAMES = INPUT_FRAMES + PRED_FRAMES  # 10 frames total needed per sequence

    # --- Model ---
    LATENT_DIM = 64  # Size of 'b' in original code

    # --- Training ---
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 200

    # Hyperparameters from original driver.py
    LAMBDA_IDENTITY = 1.0  # Reconstruction
    NU_BACKWARD = 0.5  # Backward prediction
    ETA_CONSISTENCY = 0.01  # Matrix consistency

    SEED = 42