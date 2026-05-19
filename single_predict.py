import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from get_data import universal_save
from get_data import CMixup
from get_data import plot_training_history
from model.model_mlp import build_mlp_model
from sklearn import metrics
from sklearn.metrics import r2_score

def train_model(model, X_train, Y_train, X_val, Y_val,
                epochs=100, batch_size=16,
                learning_rate=0.001, min_delta=0.0001, patience=100,
                save_path='models'):
    """
        Args:
            MLP model for concentration prediction
            X_train, Y_train: Reconstructed spectra and reference concentration
            X_val, Y_val: Reconstructed spectra and reference concentration
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
            min_delta: Minimum improvement threshold for early stopping
            patience: Patience for early stopping (number of epochs without improvement)
            save_path: Directory to save model weights and training results

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

    cmixup = CMixup(bandwidth=1.0, distance_metric='l2', beta_alpha=0.4)

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

        train_preds = []
        train_targets = []
        for batch_x, batch_y in train_dataset:
            mixed_x, mixed_y = cmixup(batch_x, batch_y)
            loss_value, preds = train_step(model, optimizer, loss_fn, mixed_x, mixed_y)
            epoch_train_loss += loss_value
            train_preds.extend(preds)
            train_targets.extend(mixed_y)
            train_batches += 1
        train_preds_cat = tf.concat(train_preds, axis=0)
        train_truth_cat = tf.concat(train_targets, axis=0)
        avg_train_loss = epoch_train_loss / train_batches
        avg_train_r2 = r2_score(train_truth_cat.numpy(), train_preds_cat.numpy())

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
        history['train_r2'].append(avg_train_r2)
        history['val_r2'].append(avg_val_r2)

        improvement_msg = ""
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_weights = model.get_weights()
            patience_counter = 0
            best_epoch = epoch + 1
            improvement_msg = f"best (Val Loss: {best_val_loss:.6f})"

            model.save_weights(f'{save_path}/best_model.weights.h5')
        else:
            patience_counter += 1
            improvement_msg = f"stop number: {patience_counter}/{patience}"

        if (epoch + 1) % 1 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, Train R2: {avg_train_r2:.4f} '
                f'Val Loss: {avg_val_loss:.4f}, Val R2: {avg_val_r2:.4f}, '
                f'{improvement_msg}')

        if patience_counter >= patience:
            print(f"stop training at epoch {epoch + 1}")
            print(f"best model at epoch {best_epoch}，val_loss: {best_val_loss:.6f}")
            break

    if best_model_weights is not None:
        model.set_weights(best_model_weights)

    print("\nsave final model...")
    model.save_weights(f'{save_path}/final_model.weights.h5')

    history.update({
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_r2': history['val_r2'][best_epoch - 1] if best_epoch > 0 else 0,
        'stopped_early': patience_counter >= patience,
    })

    return history


def test(model, ratio, test_data):
    pred_test = model.predict(test_data["data"]) * ratio

    MSE = metrics.mean_squared_error(test_data["labels"], pred_test)
    RMSE = np.sqrt(metrics.mean_squared_error(test_data["labels"], pred_test))
    R2 = metrics.r2_score(test_data["labels"], pred_test)
    print('MSE:', MSE)
    print('RMSE:', RMSE)
    print('R2:', R2)

    plot_test = pred_test
    plot_target = test_data["labels"]
    s1 = plt.scatter(plot_target, plot_test, marker='x', linewidth=5.0)
    s2 = plt.scatter(plot_target, plot_target, color='red', linewidth=2.0, linestyle='--')
    plt.legend((s1, s2), ("Predicted concentration", "Reference concentration"), fontsize=25)
    plt.title('{} {}'.format(' R2 = ', '%.6f' % (R2)), fontsize=25)
    plt.xlabel('Reference concentration', fontsize=25)
    plt.ylabel('Predicted concentration', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()


def main():
    """
        Main training pipeline for CH4 concentration prediction.

        Steps:
        1. Load data
        2. Prepare data
        3. Build MLP model architecture
        4. Train model with CMixup augmentation
        5. Save model, training history and configuration
        6. Evaluate model on test data
    """
    with open("result/CH4/20241216/mixup_cnn_eca_lstm/reconstruct_train.pkl", 'rb') as f:  # read
        train_data = pickle.load(f)
    with open("result/CH4/20241216/mixup_cnn_eca_lstm/reconstruct_val.pkl", 'rb') as f:  # read
        val_data = pickle.load(f)
    with open("result/CH4/20241216/mixup_cnn_eca_lstm/reconstruct_test.pkl", 'rb') as f:  # read
        test_data = pickle.load(f)

    train = train_data["data"]
    train_target = train_data["labels"]

    val = val_data["data"]
    val_target = val_data["labels"]

    ratio = 10000
    X_train = np.float32(train)
    Y_train = np.float32(train_target)/ratio
    X_val = np.float32(val)
    Y_val = np.float32(val_target)/ratio
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)

    input_shape = (X_train.shape[1], )
    hidden_units = [256, 128, 64]
    out_dim = Y_train.shape[1]
    dropout_rate = 0.5
    l2_reg = 1e-4
    model = build_mlp_model(input_shape=input_shape,
                            hidden_units=hidden_units,
                            out_dim=out_dim,
                            dropout_rate=dropout_rate,
                            l2_reg=l2_reg)
    print(model.summary())
    epochs = 500
    batch_size = 64
    learning_rate = 0.001
    min_delta = 0.0001
    patience = 20
    save_path = 'result/CH4/20241216/mixup_cnn_eca_lstm/concentration'

    start_time = time.time()
    history = train_model(model, X_train, Y_train, X_val, Y_val,
                epochs=epochs, batch_size=batch_size,
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
        'hidden_units': hidden_units,
        'out_dim': out_dim,
        'dropout_rate': dropout_rate,
        'l2_reg': l2_reg,
    }

    with open(f'{save_path}/network_config', 'w', encoding='utf-8') as f:
        json.dump(network_config, f, indent=2, ensure_ascii=False)

    plot_training_history(history)

    test(model, ratio, test_data)

if __name__ == '__main__':
    main()













