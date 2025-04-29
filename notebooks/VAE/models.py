import jax.numpy as jnp
from jax import random, grad, vmap, pmap
from jax.lax import pmean
from flax import linen as nn
from flax import jax_utils
from flax import struct
from flax.training import train_state
import optax

from functools import partial
from typing import Callable

import archs

class TrainState(train_state.TrainState):
    encode_fn: Callable = struct.field(pytree_node=False)
    decode_fn: Callable = struct.field(pytree_node=False)

def _create_encoder_arch(config):
    if config.name == 'MlpEncoder':
        arch = archs.MlpEncoder(**config)
    else:
        raise NotImplementedError(
            f'Arch {config.name} not supported yet!')
    return arch

def _create_decoder_arch(config):
    if config.name == 'MlpDecoder':
        arch = archs.MlpDecoder(**config)
    elif config.name == 'Mlp':
        arch = archs.Mlp(**config)
    else:
        raise NotImplementedError(
            f'Arch {config.name} not supported yet!')
    return arch


def _create_optimizer(config):
    if config.optimizer == 'Adam':
        lr = optax.exponential_decay(init_value=config.learning_rate,
                                     transition_steps=config.decay_steps,
                                     decay_rate=config.decay_rate)
        optimizer = optax.adam(learning_rate=lr, b1=config.beta1, b2=config.beta2,
                               eps=config.eps)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


class VAE(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(self, x, eps):
        z, _ = self.encoder(x, eps)
        x = self.decoder(z)
        return x

    def _encode(self, x, eps):
        z, kl_loss = self.encoder(x, eps)
        return z, kl_loss

    def _decode(self, z):
        x = self.decoder(z)
        return x
    
def _create_train_state(config):
    # Build architecture
    encoder = _create_encoder_arch(config.encoder_arch)
    decoder = _create_decoder_arch(config.decoder_arch)
    arch = VAE(encoder, decoder)
    # Initialize params
    x = jnp.ones(config.input_dim)
    eps = jnp.ones(config.eps_dim)
    key = random.PRNGKey(config.seed)
    params = arch.init(key, x, eps)
    print(arch.tabulate(key, x, eps))
    # Vectorized functions across a mini-batch
    apply_fn = vmap(arch.apply, in_axes=(None,0,0))
    encode_fn = vmap(lambda params, x, eps: arch.apply(params, x, eps, method=arch._encode), in_axes=(None,0,0))
    decode_fn = vmap(lambda params, z: arch.apply(params, z, method=arch._decode), in_axes=(None,0))
    # Optimizaer
    tx = _create_optimizer(config.optim)
    # Create state
    state = TrainState.create(apply_fn=apply_fn,
                                params=params,
                                tx=tx,
                                encode_fn=encode_fn,
                                decode_fn=decode_fn)
    # Replicate state across devices
    state = jax_utils.replicate(state) 
    return state


# Define the model
class VariationalAutoencoder:
    def __init__(self, config): 
        self.config = config
        self.state = _create_train_state(config)
        self.beta = config.beta

    # Computes KL loss across a mini-batch
    def kl_loss(self, params, x, eps):
        _, loss = self.state.encode_fn(params, x, eps)
        return jnp.mean(loss)

    # Computes reconstruction loss across a mini-batch for a single MC sample
    def recon_loss(self, params, x, eps):
        outputs = self.state.apply_fn(params, x, eps)
        loss = jnp.mean((x-outputs)**2)
        return loss
    
    # Computes total loss across a mini-batch for multiple MC samples
    def loss(self, params, batch):
        x, eps = batch
        kl_loss = vmap(self.kl_loss, in_axes=(None,None,0))(params, x, eps)
        recon_loss = vmap(self.recon_loss, in_axes=(None,None,0))(params, x, eps)
        kl_loss = jnp.mean(kl_loss)
        recon_loss = jnp.mean(recon_loss)
        loss = self.beta*kl_loss + recon_loss
        return loss
    
    @partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(0,))
    def eval_losses(self, params, batch):
        x, eps = batch
        kl_loss = vmap(self.kl_loss, in_axes=(None,None,0))(params, x, eps)
        recon_loss = vmap(self.recon_loss, in_axes=(None,None,0))(params, x, eps)
        kl_loss = jnp.mean(kl_loss)
        recon_loss = jnp.mean(recon_loss)
        return kl_loss, recon_loss

    # Define a compiled update step
    @partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(0,))
    def step(self, state, batch):
        grads = grad(self.loss)(state.params, batch)
        grads = pmean(grads, 'num_devices')
        state = state.apply_gradients(grads=grads)
        return state