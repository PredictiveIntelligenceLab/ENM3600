import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Iterator, Dict, Tuple, Any


class MNISTVAEDataLoader:
    def __init__(
        self, 
        batch_size: int = 32, 
        num_mc_samples: int = 1,
        input_dim: int = 784,
        latent_dim: int = 20,
        shuffle: bool = True, 
        seed: int = 42
    ):
        """
        Initialize the MNIST VAE data loader compatible with multi-GPU execution.
        
        Args:
            batch_size: Global batch size (will be divided among devices)
            num_mc_samples: Number of Monte Carlo samples for the VAE
            input_dim: Dimension of input images (flattened)
            latent_dim: Dimension of the latent space
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
        """
        self.num_devices = jax.device_count()
        # Ensure the batch size is divisible by the number of devices
        assert batch_size % self.num_devices == 0, f"Batch size must be divisible by device count ({self.num_devices})"
        self.global_batch_size = batch_size
        # Per-device batch size
        self.per_device_batch_size = batch_size // self.num_devices
        self.num_mc_samples = num_mc_samples
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.shuffle = shuffle
        self.seed = seed
        
        print(f"JAX running on {self.num_devices} devices")
        print(f"Global batch size: {self.global_batch_size}")
        print(f"Per-device batch size: {self.per_device_batch_size}")

    def _load_dataset(self):
        """Load the MNIST dataset from TensorFlow Datasets."""
        dataset, info = tfds.load(
            'mnist',
            split='train',
            as_supervised=True,
            shuffle_files=self.shuffle,
            with_info=True
        )
        
        return dataset, info
        
    def get_iterator(self, rng_key: jnp.ndarray = None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Returns an iterator that yields batches shaped for VAE training with pmap.
        
        Args:
            rng_key: JAX random key for generating noise samples
            
        Returns:
            Iterator yielding tuples of (images, labels, eps) with shapes:
            - images: (num_devices, batch_size_per_device, input_dim)
            - labels: (num_devices, batch_size_per_device)
            - eps: (num_devices, num_mc_samples, batch_size_per_device, latent_dim)
        """
        if rng_key is None:
            rng_key = random.PRNGKey(self.seed)
            
        # Load MNIST dataset
        ds, info = self._load_dataset()
        
        # Get total number of examples
        self.num_examples = info.splits['train'].num_examples
        
        # Define preprocessing function
        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
            image = tf.reshape(image, (-1,))  # Flatten from (28, 28, 1) to (784,)
            return image, label
        
        # Apply preprocessing
        ds = ds.map(preprocess)
        
        # Create repeating dataset that will never end
        if self.shuffle:
            ds = ds.shuffle(buffer_size=10000, seed=self.seed)
        
        # Use global batch size for TF dataset
        ds = ds.batch(self.global_batch_size).prefetch(tf.data.AUTOTUNE).repeat()
        
        # Create the iterator
        tf_iterator = iter(ds)
        
        # Create a JAX iterator wrapper that reshapes for pmap and adds noise
        def jax_iterator():
            nonlocal rng_key
            while True:
                # Get next batch from TensorFlow dataset
                images, labels = next(tf_iterator)
                
                # Convert TF tensors to numpy arrays
                images_np = images.numpy()
                labels_np = labels.numpy()
                
                # Reshape for pmap: (num_devices, per_device_batch_size, ...)
                images_reshaped = images_np.reshape(self.num_devices, self.per_device_batch_size, self.input_dim)
                labels_reshaped = labels_np.reshape(self.num_devices, self.per_device_batch_size)
                
                # Generate random noise for the VAE
                rng_key, subkey = random.split(rng_key)
                
                # Create a separate key for each device
                subkeys = random.split(subkey, self.num_devices)
                
                # Generate noise using the device-specific keys (done on CPU for simplicity)
                # Final shape: (num_devices, num_mc_samples, per_device_batch_size, latent_dim)
                eps_list = []
                for i in range(self.num_devices):
                    device_eps = random.normal(
                        subkeys[i], 
                        (self.num_mc_samples, self.per_device_batch_size, self.latent_dim)
                    )
                    eps_list.append(device_eps)
                
                eps = jnp.stack(eps_list)
                
                # Convert to JAX arrays and yield
                yield (
                    jnp.array(images_reshaped),
                    jnp.array(labels_reshaped),
                    eps
                )
        
        return jax_iterator()
    
    def get_test_data(self, rng_key: jnp.ndarray = None) -> Dict[str, jnp.ndarray]:
        """
        Returns the test dataset as JAX arrays, formatted for VAE evaluation.
        
        Args:
            rng_key: JAX random key for generating noise samples
            
        Returns:
            Dict containing test data with shapes for pmap
        """
        if rng_key is None:
            rng_key = random.PRNGKey(self.seed)
            
        test_ds = tfds.load(
            'mnist',
            split='test',
            as_supervised=True,
        )
        
        # Define preprocessing function
        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.reshape(image, (-1,))
            return image, label
            
        # Apply preprocessing
        test_ds = test_ds.map(preprocess).batch(10000)  # Load all at once
        
        # Get the single batch containing all test data
        images, labels = next(iter(test_ds))
        
        # Convert to numpy
        x_test = images.numpy()
        y_test = labels.numpy()
        
        test_size = x_test.shape[0]
        
        # We need to pad the test set to be divisible by num_devices * per_device_batch_size
        total_batch_size = self.num_devices * self.per_device_batch_size
        remainder = test_size % total_batch_size
        
        if remainder != 0:
            # Calculate padding needed
            pad_size = total_batch_size - remainder
            
            # Pad the arrays
            x_test = np.pad(x_test, ((0, pad_size), (0, 0)), mode='constant')
            y_test = np.pad(y_test, ((0, pad_size),), mode='constant')
            
            # Update test size
            test_size = x_test.shape[0]
        
        # Calculate number of batches
        num_batches = test_size // total_batch_size
        
        # Reshape for pmap
        x_test = x_test.reshape(num_batches, self.num_devices, self.per_device_batch_size, self.input_dim)
        y_test = y_test.reshape(num_batches, self.num_devices, self.per_device_batch_size)
        
        # Generate random noise for all test batches
        rng_key, subkey = random.split(rng_key)
        eps = random.normal(
            subkey, 
            (num_batches, self.num_devices, self.num_mc_samples, self.per_device_batch_size, self.latent_dim)
        )
        
        return {
            'images': jnp.array(x_test),
            'labels': jnp.array(y_test),
            'eps': jnp.array(eps),
            'num_examples': test_size - pad_size if remainder != 0 else test_size  # Original count
        }