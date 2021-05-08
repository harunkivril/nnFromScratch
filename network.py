import numpy as np
import pickle

def softmax(X):
    shifted_exp = np.exp(X.T - np.max(X, axis=1))
    return (shifted_exp / np.sum(shifted_exp, axis=0)).T


def softmax_prime(output):
    grad_start = np.einsum('ki, kj->kij', output, output)
    identity = np.eye(grad_start.shape[1])
    identities = np.stack([identity]*grad_start.shape[0])
    output_matrix = np.stack([output]*output.shape[1], axis=2)
    return np.multiply(output_matrix, identities) - grad_start


def sigmoid(X):
    return 1/(1+np.exp(-X))


def sigmoid_prime(output):
    return np.multiply(output, (1-output))


def cross_entropy_loss(X,y):
    return -np.sum(np.multiply(y, np.log(X)))/y.shape[0]


def cross_entropy_prime(X,y):
    return -y/(X)


def encode_words(indice_matrix, n_words=250):
    indice_matrix = indice_matrix.T
    if indice_matrix.shape[0] == 1:
        return (np.arange(n_words) == indice_matrix[...,None]).astype(int)[0]    
    return (np.arange(n_words) == indice_matrix[...,None]).astype(int)


class Network:
    def __init__(self, dictionary_size, embedding_size, hidden_layer_size, nwords=3, sigma=0.01, init_weights=True):
        self.dict_size = dictionary_size
        self.e_size = embedding_size
        self.h_size = hidden_layer_size
        self.n_words = nwords
        
        if init_weights:
            np.random.seed(3136)
            self.W1 = np.random.normal(scale=sigma,size=(dictionary_size, embedding_size))

            self.W2 = np.random.normal(scale=sigma, size=(embedding_size*nwords, hidden_layer_size))
            self.bias1 = np.random.normal(scale=sigma, size=(1,hidden_layer_size))

            self.W3 = np.random.normal(scale=sigma,size=(hidden_layer_size, dictionary_size))
            self.bias2 = np.random.normal(scale=sigma, size=(1,dictionary_size))
        
    def forward(self,words):
        
        self.word1, self.word2, self.word3 = words
        
        self.e1 = np.matmul(self.word1, self.W1)
        self.e2 = np.matmul(self.word2, self.W1)
        self.e3 = np.matmul(self.word3, self.W1)
        
        self.e = np.concatenate([self.e1, self.e2, self.e3], axis=1)
        
        self.h = np.matmul(self.e, self.W2) + self.bias1
        self.h = sigmoid(self.h)
        
        self.output = np.matmul(self.h, self.W3) + self.bias2
        self.output = softmax(self.output)
        
        return self.output
        
    def backward(self, y):
        X = self.output
        batch_size = X.shape[0]
        
        #self.loss_grad  = cross_entropy_prime(X, y) #bs x 250
        #self.softmax_grad = softmax_prime(X) # 250 x 250
        #delta_3 =  np.einsum("ki,kij->kj", self.loss_grad, self.softmax_grad) # bs x 250
        
        # To do things faster we can benefit from CE softmax derivative relation
        delta_3 = X-y
       
        self.W3_grad = np.matmul(delta_3.T, self.h).T / batch_size
        self.bias2_grad = np.mean(delta_3, axis=0)
        
        sigmoid_grad = sigmoid_prime(self.h)
        delta_2 = np.multiply(np.dot(delta_3, self.W3.T), sigmoid_grad)
        
        self.W2_grad = np.matmul(delta_2.T, self.e).T / batch_size
        self.bias1_grad = np.mean(delta_2, axis=0)
        
        
        delta_1 = np.matmul(delta_2, self.W2.T)
        
        delta_1_1 = delta_1[:,:self.e_size]
        delta_1_2 = delta_1[:,self.e_size:self.e_size*2 ]
        delta_1_3 = delta_1[:,self.e_size*2: ]
        
        
        W1_grad1 = np.matmul(delta_1_1.T, self.word1).T / batch_size

        W1_grad2 = np.matmul(delta_1_2.T, self.word2).T / batch_size
        
        W1_grad3 = np.matmul(delta_1_3.T, self.word3).T / batch_size
        
        self.W1_grad = W1_grad1 + W1_grad2 + W1_grad3
    
    def save_network(self, save_path):
        
        parameters = {}
        parameters["W1"] = self.W1
        
        parameters["W2"] = self.W2
        
        parameters["W3"] = self.W3
        
        parameters["bias2"] = self.bias2
        parameters["bias1"] = self.bias1
        
        with open(save_path, "wb") as file:
            pickle.dump(parameters, file)
            
    @classmethod
    def load_network(cls, model_path):
        
        with open(model_path, "rb") as file:
            parameters = pickle.load(file)
        
        cls.W1 = parameters["W1"]
        
        cls.W2 = parameters["W2"]
        
        cls.W3 = parameters["W3"]
        
        cls.bias2 = parameters["bias2"]
        cls.bias1 = parameters["bias1"]
        
        dict_size = parameters["W1"].shape[0]
        embedding_size = parameters["W1"].shape[1]
        hidden_size = parameters["W2"].shape[1]
        n = int(parameters["W2"].shape[0]/embedding_size)
        
        return cls(dict_size, embedding_size, hidden_size, nwords=n, init_weights=False)
           
        
    def SGD_step(self, lr=0.01):
        self.W1 -= lr*self.W1_grad
        self.W2 -= lr*self.W2_grad
        self.W3 -= lr*self.W3_grad
        
        self.bias1 -= lr*self.bias1_grad
        self.bias2 -= lr*self.bias2_grad
        
