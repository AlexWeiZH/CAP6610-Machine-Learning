from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.datasets import mnist
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

input_dim = 784
latent_dim = 2

# Define encoder network
encoder_inputs = layers.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(encoder_inputs)
x = layers.Dense(128, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Define sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5*z_log_var) * epsilon
z = Lambda(sampling)([z_mean, z_log_var])

# Define decoder network
decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(decoder_inputs)
x = layers.Dense(256, activation='relu')(x)
decoder_outputs = layers.Dense(784, activation='sigmoid')(x)

# Define VAE model
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = Model(encoder_inputs, vae_outputs, name='vae')


# Define VAE loss function
reconstruction_loss = mse(encoder_inputs, vae_outputs)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')


# Load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), input_dim))
x_test = np.reshape(x_test, (len(x_test), input_dim))

vae.fit(x_train, epochs=50, batch_size=128, validation_data=(x_test, None))

# Generate new images from the trained VAE
# Sample from the latent space
n_samples = 10
z_sample = np.random.normal(size=(n_samples, latent_dim))

# Decode the samples
x_decoded = decoder.predict(z_sample)

# Reshape the decoded images to their original shape
x_decoded = x_decoded.reshape(n_samples, 28, 28)

# Plot the decoded images
plt.figure(figsize=(10, 2))
for i in range(n_samples):
    # Display the original image and the decoded image
    ax = plt.subplot(2, n_samples, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.gray()

    ax = plt.subplot(2, n_samples, i + n_samples + 1)
    plt.imshow(x_decoded[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

