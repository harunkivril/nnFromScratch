import network as nn
import numpy as np

MODEL_PATH = "./model.pkl"

X_train = np.load('./data/train_inputs.npy')
X_train = nn.encode_words(X_train)
y_train = np.load('./data/train_targets.npy').reshape(-1,1)
y_train = nn.encode_words(y_train)

X_val = np.load('./data/valid_inputs.npy')
X_val = nn.encode_words(X_val)
y_val = np.load('./data/valid_targets.npy').reshape(-1,1)
y_val = nn.encode_words(y_val)

X_test = np.load('./data/test_inputs.npy')
X_test = nn.encode_words(X_test)
y_test = np.load('./data/test_targets.npy').reshape(-1,1)
y_test = nn.encode_words(y_test)

vocab = list(np.load('./data/vocab.npy'))

network = nn.Network.load_network(MODEL_PATH)

city_of_new = 'city of new'
city_of_new_idx = [vocab.index(x) for x in city_of_new.split(' ')]

life_in_the = 'life in the'
life_in_the_idx = [vocab.index(x) for x in life_in_the.split(' ')]

he_is_the = 'he is the'
he_is_the_idx = [vocab.index(x) for x in he_is_the.split(' ')]

train_pred = network.forward(X_train)
valid_pred = network.forward(X_val)
test_pred = network.forward(X_test)

train_loss = nn.cross_entropy_loss(train_pred, y_train)
valid_loss = nn.cross_entropy_loss(valid_pred, y_val)
test_loss = nn.cross_entropy_loss(test_pred, y_test)

train_acc = sum(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))/y_train.shape[0]
valid_acc = sum(np.argmax(valid_pred, axis=1) == np.argmax(y_val, axis=1))/y_val.shape[0]
test_acc = sum(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))/y_test.shape[0]

batch = np.array([city_of_new_idx, life_in_the_idx, he_is_the_idx])
encoded_batch = nn.encode_words(batch)
pred_batch = network.forward(encoded_batch)

top5 = np.argsort(-pred_batch, axis=1)[:, :5]
probs = pred_batch[np.arange(pred_batch.shape[0])[:, None], top5]

print(f'Train Loss: {train_loss},       Train Accuracy: {train_acc}')
print(f'Validation Loss: {valid_loss},  Validation Accuracy: {valid_acc}')
print(f'Test Loss: {test_loss},          Test Accuracy: {test_acc}')

print("Predictions to examples:")
print(f"    {city_of_new} {vocab[top5[0,0]]}, P: {probs[0,0]}")
print(f"    {life_in_the} {vocab[top5[1,0]]}, P: {probs[1,0]}")
print(f"    {he_is_the} {vocab[top5[2,0]]},   P: {probs[2,0]}")

for i, sentence in enumerate([city_of_new, life_in_the, he_is_the]):
    print("\n",sentence,":")
    print("  Words ,  Probs")
    print(np.array([[vocab[idx] for idx in top5[i]], probs[i]]).T)
