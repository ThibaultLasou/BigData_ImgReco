#!/home/tlasou/anaconda3/bin/python3
"""
#!/opt/anaconda3/bin/python3
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
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

def correctRatio(net, data, label):
    predicted = predict(net, data)
    nbCorrect = 0
    for i in range(0,len(predicted)):
        if predicted[i] == label.data[i]:
            nbCorrect += 1
    ratio = float(nbCorrect)/float(len(predicted))*100
    print(str(nbCorrect) + "/" + str(len(predicted)) + "=>" + str(ratio))
    return ratio

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.learning_rate = 1e-3

    def forward(self, x, training=False):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))

        return x

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(4*4*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.learning_rate = 1e-3

    def forward(self, x, training=False):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))

        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)

        self.learning_rate = 1e-3

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x

def initNet(netType):
    if netType == 'CNN':
        return CNN()
    if netType == 'MLP':
        return MLP()
    if netType == 'LeNet':
        return LeNet()
    else:
        sys.exit("Unknown network type")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Arguments : Network type (CNN, MLP, LeNet)")
    # Hyperparameters
    epoch_nbr = 100
    batch_size = 100
#    learning_rate = 1e-2

    # Data loading
    X0 = Variable(torch.from_numpy(np.load("data/trn_img.npy")).type(torch.FloatTensor))
    lbl0 = Variable(torch.from_numpy(np.load("data/trn_lbl.npy")).type(torch.FloatTensor).long())
#    X0 = Variable(torch.from_numpy(np.load("data/trn_img.npy"))[:100].type(torch.FloatTensor))
#    lbl0 = Variable(torch.from_numpy(np.load("data/trn_lbl.npy"))[:100].type(torch.FloatTensor).long())
    X1 = Variable(torch.from_numpy(np.load("data/dev_img.npy")).type(torch.FloatTensor))
    lbl1 = Variable(torch.from_numpy(np.load("data/dev_lbl.npy")).type(torch.FloatTensor).long())


    net = initNet(sys.argv[1])
    optimizer = optim.SGD(net.parameters(), lr=net.learning_rate)
    train_correct = []
    dev_correct = []
    begin = time.time()
    for e in range(epoch_nbr):
        print("Epoch", e)
        for i in range(0, X0.data.size(0), batch_size):
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(X0[i:i+batch_size])
            loss = F.nll_loss(predictions_train, lbl0[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update
        if e%10 == 9:
            train_correct.append(correctRatio(net, X0, lbl0))
            dev_correct.append(correctRatio(net, X1, lbl1))
    end = time.time()
    print("Training took " + str(int(end-begin)) + "s")
    plt.plot(range(10,epoch_nbr+1,10), train_correct, label="train")
    plt.plot(range(10,epoch_nbr+1,10), dev_correct, label="dev")
    plt.legend()
    plt.show()
