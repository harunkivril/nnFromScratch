import network as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
from time import time
import os

BATCH_SIZES = (16, 128, 512)
MAX_EPOCHS = 1
LRs= (0.1, 0.01,1)

checkpoint_paths = {}
max_no_improvement = 8

X_train = np.load('./data/train_inputs.npy')
y_train = np.load('./data/train_targets.npy').reshape(-1,1)

X_val = np.load('./data/valid_inputs.npy')
y_val = np.load('./data/valid_targets.npy').reshape(-1,1)

indices = list(range(X_train.shape[0]))
np.random.seed(3136)
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

best_models = {}
for batch_size in BATCH_SIZES:
    for lr in LRs:
        save_name = f"bs_{batch_size}_lr_{lr}"
        print(f"Training for parameters: Batch Size:{batch_size}, lr: {lr}")

        n_batches = X_train.shape[0]//batch_size + ((X_train.shape[0]%batch_size !=0))

        if not checkpoint_paths.get(save_name) is None:
            network= nn.Network.load_network(checkpoint_paths.get(save_name))
            print("Checkpoint Loaded...")
        else:
            network = nn.Network(250,16,128)

        X_valid =  nn.encode_words(X_val)
        y_valid =  nn.encode_words(y_val)
        
        early_stop = 0
        best_acc = 0
        train_losses = []
        validation_losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(MAX_EPOCHS):
            start_time = time()
            total_loss = 0
            correct_words = 0
            for i in range(n_batches):
                start = i*batch_size
                end = (i+1)*batch_size

                X_batch = X_train[start:end]
                X_batch = nn.encode_words(X_batch)

                y_batch = y_train[start:end]
                y_batch = nn.encode_words(y_batch)
                
                y_pred = network.forward(X_batch)
                network.backward(y_batch)
                
                network.SGD_step(lr)

                batch_loss = nn.cross_entropy_loss(y_pred, y_batch)
                total_loss += batch_loss
                correct_words += sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))


                if (i % 1000) == 0:
                    print(f"Epoch: {epoch}, Batch: {i} Train loss: {batch_loss}")

            train_acc = correct_words/X_train.shape[0]
            train_accuracy.append(train_acc)
            train_losses.append(total_loss/n_batches)

            print(f"Time: {time()-start_time} Validating Epoch {epoch}...")
            y_pred = network.forward(X_valid)
            loss = nn.cross_entropy_loss(y_pred, y_valid)
            validation_losses.append(loss)
            correct = sum(np.argmax(y_pred, axis=1) == np.argmax(y_valid, axis=1))
            test_acc = correct/y_valid.shape[0]
            validation_accuracy.append(test_acc)
            print(f"Epoch {epoch}: Train loss: {train_losses[-1]}, Train accuracy:{train_acc}, Valid loss: {loss}, Valid accuracy: {test_acc}")
            if not os.path.exists(f'./models/{save_name}/'):
                os.makedirs(f"./models/{save_name}")
            network.save_network(f'./models/{save_name}/Epoch{epoch}.pkl')
            early_stop +=1
            if test_acc >= best_acc:
                early_stop = 0
                best_acc = test_acc
                network.save_network(f'./models/{save_name}/BestModel.pkl')
            if early_stop > max_no_improvement:
                break

        metrics = {}
        metrics["train_accuracy"] = train_accuracy
        metrics["validation_accuracy"] = validation_accuracy
        metrics["train_loss"] = train_losses
        metrics["validation_loss"] = validation_losses
        metrics["batch_size"] = batch_size
        metrics["learning_rate"] = lr

        with open(f"./models/summary_{save_name}.pkl", "wb") as file:
            pickle.dump(metrics, file)

        plt.style.use("ggplot")
        plt.figure(figsize=(12,8))
        plt.plot(list(range(len(train_accuracy))), train_accuracy, label='Train Accuracy')
        plt.plot(list(range(len(validation_accuracy))), validation_accuracy, label='Validation Accuracy')
        plt.xlabel("# of Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"./models/{save_name}/accuracy_plot.png")

        plt.figure(figsize=(12,8))
        plt.plot(list(range(len(train_losses))), train_losses, label='Train Loss')
        plt.plot(list(range(len(validation_losses))), validation_losses, label='Validation Loss')
        plt.xlabel("# of Epoch")
        plt.ylabel("CE Loss")
        plt.legend()
        plt.savefig(f"./models/{save_name}/loss_plot.png")

        best_models[save_name] = best_acc

with open("./models/best_models.pkl", "wb") as file:
    pickle.dump(best_models, file)
print(best_models)
print("DONE")



