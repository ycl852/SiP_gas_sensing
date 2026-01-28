import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np

def universal_save(data, filename):
    with open(filename, 'w') as f:
        if isinstance(data, (list, tuple, np.ndarray)):
            for item in data:
                f.write(f"{item}\n")
        else:
            f.write(str(data))

def save_to_pickle2(data_tuple, save_path):
    data_dict = {'data': data_tuple[0],
                'labels': data_tuple[1]}

    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"✓ save in {save_path}")


def save_to_pickle3(data_tuple, save_path):
    data_dict = {'data': data_tuple[0],
                'target': data_tuple[1],
                'labels': data_tuple[2]}

    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"✓ save in {save_path}")


class CMixup:
    def __init__(self, bandwidth=1.0, distance_metric='l2', beta_alpha=0.4):
        self.bandwidth = bandwidth
        self.distance_metric = distance_metric
        self.beta_alpha = beta_alpha

    def _compute_pairwise_distance(self, y):

        y_i = tf.expand_dims(y, axis=1)
        y_j = tf.expand_dims(y, axis=0)

        if self.distance_metric == 'l1':
            distances = tf.reduce_sum(tf.abs(y_i - y_j), axis=-1)
        else:
            distances = tf.reduce_sum(tf.square(y_i - y_j), axis=-1)

        return distances

    def _compute_kde_probabilities(self, distances):

        bandwidth_sq = tf.constant(self.bandwidth ** 2, dtype=distances.dtype)
        similarities = tf.exp(-distances / (2.0 * bandwidth_sq))

        batch_size = tf.shape(similarities)[0]
        mask = 1.0 - tf.eye(batch_size, dtype=similarities.dtype)
        similarities = similarities * mask

        probs = similarities / (tf.reduce_sum(similarities, axis=1, keepdims=True) + 1e-8)

        return probs

    def _sample_pairs(self, probs: tf.Tensor) -> tf.Tensor:

        batch_size = tf.shape(probs)[0]
        log_probs = tf.math.log(probs + 1e-8)
        indices = tf.random.categorical(log_probs, num_samples=1)
        indices = tf.squeeze(indices, axis=-1)

        return indices

    def _sample_mixing_weights(self, batch_size: int) -> tf.Tensor:

        if self.beta_alpha <= 0:
            lam = tf.random.uniform((batch_size, 1), minval=0.0, maxval=1.0)
        else:

            alpha = tf.constant(self.beta_alpha, dtype=tf.float32)
            u = tf.random.uniform((batch_size, 1), minval=0.0, maxval=1.0)
            v = tf.random.uniform((batch_size, 1), minval=0.0, maxval=1.0)
            lam = tf.math.pow(u, 1.0 / alpha) / (tf.math.pow(u, 1.0 / alpha) + tf.math.pow(v, 1.0 / alpha))

        lam = tf.clip_by_value(lam, 0.0, 1.0)

        return lam

    def __call__(self, x, y):
        batch_size = tf.shape(x)[0]
        distances = self._compute_pairwise_distance(y)
        probs = self._compute_kde_probabilities(distances)
        pair_indices = self._sample_pairs(probs)
        lam = self._sample_mixing_weights(batch_size)

        x_paired = tf.gather(x, pair_indices)
        y_paired = tf.gather(y, pair_indices)

        ndim_x = len(x.shape)
        lam_x = tf.reshape(lam, [batch_size] + [1] * (ndim_x - 1))
        lam_y = lam

        x_mixed = lam_x * x + (1.0 - lam_x) * x_paired
        y_mixed = lam_y * y + (1.0 - lam_y) * y_paired

        x_mixed = tf.cast(x_mixed, x.dtype)
        y_mixed = tf.cast(y_mixed, y.dtype)

        return x_mixed, y_mixed


class SequenceCMixup:
    def __init__(self, alpha=2.0, sigma=1.0, distance_metric="euclidean"):
        self.alpha = alpha
        self.sigma = sigma
        self.distance_metric = distance_metric

    def compute_distance_matrix(self, Y):
        batch_size = Y.shape[0]

        if self.distance_metric == "euclidean":
            Y_expanded1 = tf.expand_dims(Y, 1)
            Y_expanded2 = tf.expand_dims(Y, 0)
            distances = tf.norm(Y_expanded1 - Y_expanded2, ord='euclidean', axis=2)

        else:
            Y_norm = tf.nn.l2_normalize(Y, axis=1)
            cosine_sim = tf.matmul(Y_norm, Y_norm, transpose_b=True)
            distances = 1.0 - cosine_sim

        eye_mask = 1.0 - tf.eye(batch_size, dtype=distances.dtype)
        distances = distances * eye_mask
        return distances

    def _sample_lambda(self, batch_size, dtype=tf.float32):
        shape = [batch_size]
        one = tf.ones(shape, dtype=dtype)
        alpha_tensor = tf.cast(self.alpha, dtype)

        gamma_1 = tf.random.gamma(shape, alpha=alpha_tensor, dtype=dtype)
        gamma_2 = tf.random.gamma(shape, alpha=alpha_tensor, dtype=dtype)
        return gamma_1 / (gamma_1 + gamma_2)

    def __call__(self, X, Y):
        dtype = X.dtype
        batch_size = tf.shape(Y)[0]

        distances = self.compute_distance_matrix(Y)
        similarities = tf.exp(-distances / (2.0 * self.sigma ** 2))

        lambda_vals = self._sample_lambda(batch_size, dtype=dtype)
        lambda_vals = tf.reshape(lambda_vals, [-1, 1])

        eye_mask = tf.eye(batch_size, dtype=similarities.dtype)
        similarities = similarities * (1.0 - eye_mask)

        row_sums = tf.reduce_sum(similarities, axis=1, keepdims=True)
        probabilities = similarities / (row_sums + 1e-8)

        sampled_indices = tf.random.categorical(tf.math.log(probabilities), num_samples=1)

        paired_X = tf.gather(X, sampled_indices[:, 0])
        paired_Y = tf.gather(Y, sampled_indices[:, 0])

        if len(X.shape) > 2:
            lambda_shape = [-1] + [1] * (len(X.shape) - 1)
            lambda_X = tf.reshape(lambda_vals, lambda_shape)
        else:
            lambda_X = lambda_vals

        mixed_X = lambda_X * X + (1.0 - lambda_X) * paired_X
        mixed_Y = lambda_vals * Y + (1.0 - lambda_vals) * paired_Y

        return mixed_X, mixed_Y


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_r2'], label='Train R2')
    axes[1].plot(history['val_r2'], label='Val R2')
    axes[1].set_title('Training and Validation R2 Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R2 Score')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()