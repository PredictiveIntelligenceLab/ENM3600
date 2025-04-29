import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Sequence, Optional, Union, Dict


class Mlp(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = self.activation(x)
        x = nn.Dense(self.output_dim)(x)
        return x
    
class GaussianMlp(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = self.activation(x)
        mu = nn.Dense(self.output_dim)(x)
        logsigma = nn.Dense(self.output_dim)(x)
        return mu, logsigma
    
class MlpEncoder(nn.Module):
    latent_dim: int=8
    num_layers: int=2
    hidden_dim: int=64
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x, eps):
        mu, logsigma = GaussianMlp(self.num_layers, 
                                   self.hidden_dim,
                                   self.latent_dim,
                                   self.activation)(x)
        z = mu + eps*jnp.sqrt(jnp.exp(logsigma))
        kl_loss = 0.5*jnp.sum(jnp.exp(logsigma) + mu**2 - 1.0 - logsigma, axis=-1)
        return z, kl_loss
    
class MlpDecoder(nn.Module):
    output_dim: int=8
    num_layers: int=2
    hidden_dim: int=64
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x, eps):
        mu, logsigma = GaussianMlp(self.num_layers, 
                                   self.hidden_dim,
                                   self.latent_dim,
                                   self.activation)(x)
        z = mu + eps*jnp.sqrt(jnp.exp(logsigma))
        return z