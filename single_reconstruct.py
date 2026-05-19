import pickle
import json
import numpy as np
import time
import tensorflow as tf
from get_data import universal_save
from get_data import SequenceCMixup
from get_data import plot_training_history
from get_data import save_to_pickle3
from model.model_cnn_lstm_eca_mlp import build_spectrum_model
from sklearn import metrics
from sklearn.metrics import r2_score

def train_model(model, X_train, Y_train, X_val, Y_val,
                epochs=100, batch_size=16, alpha=2.0, sigma=1.0,
                learning_rate=0.001, min_delta=0.0001, patience=100,
                save_path='models'):
    """
        Args:
            CNN-ECA-LSTM model for spectral reconstruction
            X_train, Y_train: Time-domain signals and reference spectra
            X_val, Y_val: Time-domain signals and reference spectra
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            alpha, sigma: Hyperparameters for SequenceCMixup augmentation
            learning_rate: Learning rate for Adam optimizer
            min_delta: Minimum improvement threshold for early stopping
            patience: Patience for early stopping (number of epochs without improvement)
            save_path: Directory path to save model weights and checkpoints

        Returns:
            history: Dictionary containing training history
    """
    import os
    os.makedirs(save_path, exist_ok=True)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) \
        .shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    sequence_mixer = SequenceCMixup(alpha=alpha, sigma=sigma, distance_metric="euclidean")

    @tf.function
    def train_step(model, optimizer, loss_fn, x_batch, y_batch):
        with tf.GradientTape() as tape:
            outputs = model(x_batch, training=True)
            loss = loss_fn(y_batch, outputs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, outputs

    @tf.function
    def val_step(model, loss_fn, x_batch, y_batch):
        outputs = model(x_batch, training=False)
        loss = loss_fn(y_batch, outputs)
        return loss, outputs

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_weights = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        epoch_train_loss = 0.0
        train_batches = 0

        for batch_x, batch_y in train_dataset:
            mixed_x, mixed_y = sequence_mixer(batch_x, batch_y)
            loss_value, _ = train_step(model, optimizer, loss_fn, mixed_x, mixed_y)
            epoch_train_loss += loss_value
            train_batches += 1
        avg_train_loss = epoch_train_loss / train_batches

        epoch_val_loss = 0.0
        val_batches = 0
        val_preds = []
        val_targets = []
        for batch_x, batch_y in val_dataset:
            loss_value, preds = val_step(model, loss_fn, batch_x, batch_y)
            epoch_val_loss += loss_value
            val_preds.extend(preds)
            val_targets.extend(batch_y)
            val_batches += 1

        val_preds_cat = tf.concat(val_preds, axis=0)
        val_truth_cat = tf.concat(val_targets, axis=0)
        avg_val_loss = epoch_val_loss / val_batches
        avg_val_r2 = r2_score(val_truth_cat.numpy(), val_preds_cat.numpy())

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_r2'].append(avg_val_r2)

        improvement_msg = ""
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_weights = model.get_weights()
            patience_counter = 0
            best_epoch = epoch + 1
            improvement_msg = f"★ best (Val Loss: {best_val_loss:.6f})"

            model.save_weights(f'{save_path}/best_model.weights.h5')
        else:
            patience_counter += 1
            improvement_msg = f"stop number: {patience_counter}/{patience}"

        if (epoch + 1) % 1 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}, Val R2: {avg_val_r2:.4f}, '
                f'{improvement_msg}')

        if patience_counter >= patience:
            print(f"stop training at epoch {epoch + 1}")
            print(f"best model at epoch {best_epoch}，val_loss: {best_val_loss:.6f}")
            break

    if best_model_weights is not None:
        model.set_weights(best_model_weights)

    print("save final model...")
    model.save_weights(f'{save_path}/final_model.weights.h5')

    history.update({
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_r2': history['val_r2'][best_epoch - 1] if best_epoch > 0 else 0,
        'stopped_early': patience_counter >= patience,
    })

    return history


def test(model, train_data, val_data, test_data, load_path):
    pred_train = model.predict(train_data["data"])
    pred_val = model.predict(val_data["data"])
    pred_test = model.predict(test_data["data"])

    MSE = metrics.mean_squared_error(test_data["target"], pred_test)
    RMSE = np.sqrt(metrics.mean_squared_error(test_data["target"], pred_test))
    R2 = metrics.r2_score(test_data["target"], pred_test)
    print('MSE:', MSE)
    print('RMSE:', RMSE)
    print('R2:', R2)

    save_to_pickle3([pred_train, train_data["target"], train_data["labels"]], f'{load_path}/reconstruct_train.pkl')
    save_to_pickle3([pred_val, val_data["target"], val_data["labels"]], f'{load_path}/reconstruct_val.pkl')
    save_to_pickle3([pred_test, test_data["target"], test_data["labels"]], f'{load_path}/reconstruct_test.pkl')


def main():
    """
       Main training pipeline for CH4 spectral reconstruction.

       Steps:
       1. Load data
       2. Prepare data
       3. Build CNN-ECA-LSTM model architecture
       4. Train model with SequenceCMixup augmentation
       5. Save model, training history and configuration
       6. Evaluate model on test data
    """
    with open("result/CH4/20241216/train_CH4.pkl", 'rb') as f:  # read
        train_data = pickle.load(f)
    with open("result/CH4/20241216/val_CH4.pkl", 'rb') as f:  # read
        val_data = pickle.load(f)
    with open("result/CH4/20241216/test_CH4.pkl", 'rb') as f:  # read
        test_data = pickle.load(f)

    train = train_data["data"]
    train_target = train_data["target"]

    val = val_data["data"]
    val_target = val_data["target"]

    X_train = train.reshape(-1, train.shape[1], 1)
    Y_train = train_target
    X_val = val.reshape(-1, val.shape[1], 1)
    Y_val = val_target
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)

    input_shape = (X_train.shape[1], 1)
    conv_filters = [256, 128]
    kernel_sizes = [5, 2]
    use_eca = True
    lstm_units = 256
    use_dense = False
    use_dropout = False
    dense_units = [256, 128, 64]
    dropout_rate = 0.5
    out_dim = Y_train.shape[1]
    l2_reg = 1e-4
    model = build_spectrum_model(input_shape=input_shape,
                                 conv_filters=conv_filters,
                                 kernel_sizes=kernel_sizes,
                                 use_eca=use_eca,
                                 lstm_units=lstm_units,
                                 use_dense=use_dense,
                                 use_dropout=use_dropout,
                                 dense_units=dense_units,
                                 dropout_rate=dropout_rate,
                                 out_dim=out_dim,
                                 l2_reg=l2_reg)
    print(model.summary())
    epochs = 500
    batch_size = 16
    alpha = 2.0
    sigma = 1.0
    learning_rate = 0.001
    min_delta = 0.0001
    patience = 20
    save_path = 'result/CH4/20241216/mixup_cnn_eca_lstm'

    start_time = time.time()
    history = train_model(model, X_train, Y_train, X_val, Y_val,
                epochs=epochs, batch_size=batch_size, alpha=alpha, sigma=sigma,
                learning_rate=learning_rate, min_delta=min_delta, patience=patience,
                save_path=save_path)
    end_time = time.time()

    print(f'Total training time: {(end_time - start_time):.2f} Seconds')
    print(f'Average time per epoch: {(end_time - start_time) / epochs:.4f} Seconds')

    universal_save(f'Running time: {(end_time - start_time) / epochs:.4f} Seconds', f'{save_path}/time.txt')
    universal_save(history['best_epoch'], f'{save_path}/best_epoch.txt')
    universal_save(history['train_r2'], f'{save_path}/train_acc.txt')
    universal_save(history['train_loss'], f'{save_path}/train_loss.txt')
    universal_save(history['val_r2'], f'{save_path}/val_acc.txt')
    universal_save(history['val_loss'], f'{save_path}/val_loss.txt')
    network_config = {
        'input_shape': input_shape,
        'conv_filters': conv_filters,
        'kernel_sizes': kernel_sizes,
        'use_eca': use_eca,
        'lstm_units': lstm_units,
        'use_dense': use_dense,
        'use_dropout': use_dropout,
        'dense_units': dense_units,
        'dropout_rate': dropout_rate,
        'out_dim': out_dim,
        'l2_reg': l2_reg
    }
    with open(f'{save_path}/network_config', 'w', encoding='utf-8') as f:
        json.dump(network_config, f, indent=2, ensure_ascii=False)

    plot_training_history(history)

    test(model, train_data, val_data, test_data, save_path)

if __name__ == '__main__':
    main()













