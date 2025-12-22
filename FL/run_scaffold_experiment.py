import tensorflow as tf
import numpy as np
import os
import fl_utils
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
import random
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Force CPU usage for parallelism stability
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configuration
PARTITION_DIR = '/home/arjit/thesis/stage2Data/federated_partitions'
OUTPUT_DIR = '/home/arjit/thesis/FL/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_ROUNDS = 50
LOCAL_EPOCHS = 3  # Increased from 1 to 3
LEARNING_RATE_CLIENT = 0.01
LEARNING_RATE_SERVER = 0.5  # Reduced from 1.0 to 0.5

def train_client_worker(client_id, file_path, global_weights, global_c, client_c, learning_rate):
    """
    Worker function to train a client in a separate process.
    Re-initializes the client environment and model to ensure thread/process safety.
    """
    # Force CPU in worker process as well
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Load Data and Compute Class Weights
    df = pd.read_csv(file_path)
    labels = df['label'].values
    
    # Compute weights: n_samples / (n_classes * np.bincount(y))
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    weight_dict = dict(zip(np.unique(labels), class_weights))
    
    # Create a lookup tensor for efficiency
    full_class_weights = np.ones(fl_utils.NUM_CLASSES, dtype=np.float32)
    for cls, w in weight_dict.items():
        if cls < fl_utils.NUM_CLASSES:
            full_class_weights[cls] = w
            
    class_weights_tensor = tf.constant(full_class_weights, dtype=tf.float32)
    
    # Load Dataset
    dataset = fl_utils.load_client_dataset(file_path)
    
    # Create Model
    model = fl_utils.create_keras_model()
    model.set_weights(global_weights)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Convert inputs to tensors for training
    global_c_tensors = [tf.convert_to_tensor(c) for c in global_c]
    client_c_tensors = [tf.convert_to_tensor(c) for c in client_c]
    x_weights_tensors = [tf.convert_to_tensor(w) for w in global_weights]
    
    total_loss = 0.0
    num_batches = 0
    
    # Training Loop
    for _ in range(LOCAL_EPOCHS):
        for x_batch, y_batch in dataset:
            # Gather sample weights
            sample_weights = tf.gather(class_weights_tensor, y_batch)
            
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits, sample_weight=sample_weights)
            
            grads = tape.gradient(loss, model.trainable_variables)
            
            # SCAFFOLD Update: g_mod = g - c_i + c
            modified_grads = []
            for g, ci, c in zip(grads, client_c_tensors, global_c_tensors):
                if g is None:
                    modified_grads.append(None)
                else:
                    modified_grads.append(g - ci + c)
            
            optimizer.apply_gradients(zip(modified_grads, model.trainable_variables))
            
            total_loss += loss
            num_batches += 1
            
    # Calculate new c_i and delta_c_i
    K = float(num_batches) if num_batches > 0 else 1.0
    eta = learning_rate
    
    y_weights = model.get_weights()
    new_c_i = []
    delta_c_i = []
    
    for ci, c, x, y in zip(client_c_tensors, global_c_tensors, x_weights_tensors, y_weights):
        # c_i+ = c_i - c + (1 / (K * eta)) * (x - y_i)
        term = (x - y) / (K * eta)
        ci_plus = ci - c + term
        
        new_c_i.append(ci_plus.numpy())      # Convert back to numpy for transport
        delta_c_i.append((ci_plus - ci).numpy())
        
    avg_loss = (total_loss / num_batches).numpy() if num_batches > 0 else 0.0
    
    return client_id, y_weights, delta_c_i, avg_loss, new_c_i

def main():
    print("Initializing SCAFFOLD Experiment (Parallel Process Mode)...")
    
    # 1. Initialize Global Model
    global_model = fl_utils.create_keras_model()
    dummy_input = tf.zeros((1, fl_utils.NUM_FEATURES))
    global_model(dummy_input)
    
    # Compile for evaluation
    global_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    # Initialize Global State (Weights and Control Variates) as Numpy Arrays
    global_weights = global_model.get_weights()
    global_c = [np.zeros_like(w) for w in global_weights]
    
    # 2. Initialize Client State (Control Variates)
    client_files = {
        'client_1': os.path.join(PARTITION_DIR, 'client_1.csv'),
        'client_2': os.path.join(PARTITION_DIR, 'client_2.csv'),
        'client_3': os.path.join(PARTITION_DIR, 'client_3.csv'),
        'client_4': os.path.join(PARTITION_DIR, 'client_4.csv')
    }
    
    # Store client control variates (c_i) in a dictionary
    client_states = {}
    for cid in client_files:
        client_states[cid] = [np.zeros_like(w) for w in global_weights]
        
    # 3. Training Loop
    metrics_history = {'loss': [], 'accuracy': [], 'f1': [], 'macro_f1': []}
    
    # Load Global Test Set
    test_df = pd.read_csv(os.path.join(PARTITION_DIR, 'global_test.csv'))
    x_test = test_df.drop('label', axis=1).values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.int32)
    
    print(f"Starting training for {NUM_ROUNDS} rounds...")
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"--- Round {round_num} ---")
        
        client_weights_updates = []
        client_c_updates = []
        round_loss = 0.0
        
        # Prepare arguments for parallel execution
        tasks = []
        for cid, fpath in client_files.items():
            tasks.append((
                cid, 
                fpath, 
                global_weights, 
                global_c, 
                client_states[cid], 
                LEARNING_RATE_CLIENT
            ))
        
        # Randomize client execution order
        random.shuffle(tasks)
            
        # Run Parallel Training using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(client_files)) as executor:
            futures = [executor.submit(train_client_worker, *task) for task in tasks]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        # Process Results
        for cid, w_new, delta_ci, loss, new_ci in results:
            # Update client state
            client_states[cid] = new_ci
            
            # Calculate delta_y = y_i - x
            delta_y = [y - x for y, x in zip(w_new, global_weights)]
            
            client_weights_updates.append(delta_y)
            client_c_updates.append(delta_ci)
            round_loss += loss
            
        # Server Aggregation
        N = len(client_files)
        
        # Update Global Weights
        new_global_weights = []
        for i, w in enumerate(global_weights):
            sum_delta_y = np.zeros_like(w)
            for delta in client_weights_updates:
                sum_delta_y += delta[i]
            
            # Apply update: x_new = x + (eta_g / N) * sum(delta_y)
            update = (LEARNING_RATE_SERVER / N) * sum_delta_y
            new_global_weights.append(w + update)
            
        global_weights = new_global_weights
        global_model.set_weights(global_weights)
        
        # Update Global Control Variate
        new_global_c = []
        for i, c in enumerate(global_c):
            sum_delta_ci = np.zeros_like(c)
            for delta in client_c_updates:
                sum_delta_ci += delta[i]
            
            # c_new = c + (1 / N) * sum(delta_ci)
            update = (1.0 / N) * sum_delta_ci
            new_global_c.append(c + update)
            
        # Clip global control variates to [-1.0, 1.0] to prevent explosion
        global_c = [np.clip(c, -1.0, 1.0) for c in new_global_c]
        
        # Evaluation
        loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
        
        # Calculate F1 Score
        y_pred_probs = global_model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        f1 = f1_score(y_test, y_pred, average='weighted')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        metrics_history['loss'].append(loss)
        metrics_history['accuracy'].append(acc)
        metrics_history['f1'].append(f1)
        metrics_history['macro_f1'].append(macro_f1)
        
        print(f"Round {round_num}: Test Loss = {loss:.4f}, Test Accuracy = {acc:.4f}, Weighted F1 = {f1:.4f}, Macro F1 = {macro_f1:.4f}")
        
        if round_num % 10 == 0 or round_num == NUM_ROUNDS:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

    # 4. Save Results
    print("Training complete. Saving results...")
    
    # Save Metrics Plot
    plt.figure(figsize=(24, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(range(1, NUM_ROUNDS + 1), metrics_history['loss'], label='Test Loss')
    plt.title('SCAFFOLD Global Model Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(range(1, NUM_ROUNDS + 1), metrics_history['accuracy'], label='Test Accuracy', color='orange')
    plt.title('SCAFFOLD Global Model Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.plot(range(1, NUM_ROUNDS + 1), metrics_history['f1'], label='Weighted F1 Score', color='green')
    plt.title('SCAFFOLD Global Model Weighted F1')
    plt.xlabel('Round')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(range(1, NUM_ROUNDS + 1), metrics_history['macro_f1'], label='Macro F1 Score', color='red')
    plt.title('SCAFFOLD Global Model Macro F1')
    plt.xlabel('Round')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scaffold_training_metrics.png'))
    
    # Save Metrics CSV
    df_metrics = pd.DataFrame({
        'round': range(1, NUM_ROUNDS + 1),
        'loss': metrics_history['loss'],
        'accuracy': metrics_history['accuracy'],
        'f1': metrics_history['f1'],
        'macro_f1': metrics_history['macro_f1']
    })
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'training_metrics.csv'), index=False)
    
    # Save Final Model
    global_model.save(os.path.join(OUTPUT_DIR, 'global_model_scaffold.h5'))
    print(f"Model saved to {os.path.join(OUTPUT_DIR, 'global_model_scaffold.h5')}")

if __name__ == "__main__":
    main()
