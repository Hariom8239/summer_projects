# train.py

import tensorflow as tf
from data_loader import load_data
from model import create_model

# Load and preprocess data
data_dir = 'path/to/your_dataset_directory'  # Update this path to where your dataset is located
x_train, y_train, x_test, y_test, num_classes = load_data(data_dir)

# Create model
input_shape = x_train.shape[1:]
model = create_model(input_shape, num_classes)
model.summary()

# Define loss function, optimizer, and metrics
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

# Define training step function
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    train_acc_metric.update_state(y_batch, logits)
    return loss

# Define validation step function
@tf.function
def val_step(x_batch, y_batch):
    val_logits = model(x_batch, training=False)
    val_acc_metric.update_state(y_batch, val_logits)

# Training loop
epochs = 10
batch_size = 64

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Iterate over the batches of the dataset
    for step in range(len(x_train) // batch_size):
        x_batch_train = x_train[step * batch_size:(step + 1) * batch_size]
        y_batch_train = y_train[step * batch_size:(step + 1) * batch_size]
        
        loss = train_step(x_batch_train, y_batch_train)
        
        # Log every 100 steps
        if step % 100 == 0:
            print(f"Training loss at step {step}: {loss:.4f}")

    # Display metrics at the end of each epoch
    train_acc = train_acc_metric.result()
    print(f"Training accuracy over epoch: {train_acc:.4f}")
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch
    for step in range(len(x_test) // batch_size):
        x_batch_val = x_test[step * batch_size:(step + 1) * batch_size]
        y_batch_val = y_test[step * batch_size:(step + 1) * batch_size]
        
        val_step(x_batch_val, y_batch_val)
    
    val_acc = val_acc_metric.result()
    print(f"Validation accuracy: {val_acc:.4f}")
    val_acc_metric.reset_states()

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
