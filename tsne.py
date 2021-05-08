import network as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

MODEL_PATH = "./model.pkl"

vocab = list(np.load('./data/vocab.npy'))
network = nn.Network.load_network(MODEL_PATH)

encoded_words = np.identity(len(vocab))
embeddings = np.matmul(encoded_words, network.W1)

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=3136)
reduced_embeddings = tsne_model.fit_transform(embeddings)

x = reduced_embeddings[:,0]
y = reduced_embeddings[:,1]

plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.scatter(x, y)
for label, xi, yi in zip(vocab, x, y):
    plt.annotate(label, xy=(xi,yi), xytext=(0, 0), textcoords='offset points')

plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.title("TSNE Plot of Reduced Embeddings")

plt.savefig('./tsne_plot.png')
print("Plot saved to directory")

