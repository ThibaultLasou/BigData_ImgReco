#!/usr/bin/python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
torch.set_printoptions(profile="full")
torch.manual_seed(0)

def predict(net, data):
    predictions = net(data)
    _, predicted = torch.max(predictions.data, 1)
    return predicted

def errorRatio(net, data, label):
    predicted = predict(net, data)
    nbError = 0
    for i in range(0,len(predicted)):
        if predicted[i] != label.data[i]:
            nbError += 1
    ratio = float(nbError)/float(len(predicted))*100
    print(str(nbError) + "/" + str(len(predicted)) + "=>" + str(ratio))
    return ratio

class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
            x = x.view(-1, 1, 28, 28)
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.size(0), -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.log_softmax(self.fc2(x))
            
            return x


if __name__ == '__main__':
    # Hyperparameters
    epoch_nbr = 100
    batch_size = 100
    learning_rate = 1e-3

    # Data loading
    X0 = Variable(torch.from_numpy(np.load("data/trn_img.npy"))[:100].type(torch.FloatTensor))
    lbl0 = Variable(torch.from_numpy(np.load("data/trn_lbl.npy"))[:100].type(torch.FloatTensor).long())
    X1 = Variable(torch.from_numpy(np.load("data/dev_img.npy")).type(torch.FloatTensor))
    lbl1 = Variable(torch.from_numpy(np.load("data/dev_lbl.npy")).type(torch.FloatTensor).long())

    net = CNN()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    for e in range(epoch_nbr):
        print("Epoch", e)
        for i in range(0, X0.data.size(0), batch_size):
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(X0[i:i+batch_size])
            loss = F.nll_loss(predictions_train, lbl0[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update
            if e%10 == 9:
                errorRatio(net, X0, lbl0)
                errorRatio(net, X1, lbl1)
