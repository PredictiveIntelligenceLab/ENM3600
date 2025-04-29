import ml_collections
from flax import linen as nn


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.job_type = 'train'

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = 'MNIST'
    wandb.name = None
    wandb.tag = None

    # Simulation settings
    config.input_dim = (1,784)
    config.eps_dim = 16
    config.beta = 1e-5
    config.seed = 4

    # Encoder Architecture
    config.encoder_arch = encoder_arch = ml_collections.ConfigDict()
    encoder_arch.name = 'MlpEncoder'
    encoder_arch.num_layers = 3
    encoder_arch.hidden_dim = 64
    encoder_arch.latent_dim = config.eps_dim
    encoder_arch.activation = nn.gelu

    # Decoder Architecture
    config.decoder_arch = decoder_arch = ml_collections.ConfigDict()
    decoder_arch.name = 'Mlp'
    decoder_arch.num_layers = 3
    decoder_arch.hidden_dim = 128
    decoder_arch.output_dim = 784
    decoder_arch.activation: nn.gelu

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.num_mc_samples = 4
    training.num_epochs = 20

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 1000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_losses = True

    return config