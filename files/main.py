from __future__ import print_function
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset,TensorDataset
import torch
from scipy.io import loadmat


#Retrieving the local MNIST.mat file

current_dir = os.getcwd()  #current local directory saved in
mnist_path = (current_dir+'/mnist.mat')
mnist_raw = loadmat(mnist_path)     #raw MNIST file drawn from the matlab file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

testx = torch.tensor(mnist_raw['testX'])
testy = mnist_raw['testY']
trainx = torch.tensor(mnist_raw['trainX'])
trainy = mnist_raw['trainY']


trainy = torch.from_numpy(trainy).long().to(device)
testy = torch.from_numpy(testy).long().to(device)
trainy = torch.tensor(trainy)


img_rows = 28
img_col = 28

trainx = trainx.view(trainx.shape[0],1,28,28).float().to(device)
testx = testx.view(testx.shape[0],1,28,28).float().to(device)
trainy = trainy.squeeze(0)
testy = testy.squeeze(0)


#Variables:
learning_rate = 0.001
training_epochs = 10
batch_size = 10

#train= torch.utils.data.DataLoader(train = TensorDataset(train_X, train_Y)
#train_loader = DataLoader(train, batch_size=100, shuffle=True)dataset=train_dataset,batch_size = batch_size,shuffle=True)
train = TensorDataset(trainx,trainy)
train_loader = torch.utils.data.DataLoader(dataset=train,batch_size = batch_size, shuffle = True, drop_last = True)



#Convolution Neural Network function:

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
             nn.Conv2d(1,32, kernel_size=3, stride = 1, padding = 1),
             nn.ReLU(),
             nn.MaxPool2d(2)
         )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*64,10,bias = True)
        torch.nn.init.xavier_uniform(self.fc.weight)

    def forward(self,x):
         out = self.layer1(x)
         out = self.layer2(out)
         out = out.view(out.size(0),-1)
         out = self.fc(out)
         return out


model = CNN().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)        #Optimizer used as 'Adam'



#training

total_batch = len(train_loader)

for epoch in range(training_epochs):
    avg_cost = 0


    for X,Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)

        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost/total_batch

    print('[Epoch: {}] cost = {}'.format(epoch+1,avg_cost))

print('Learning Finished!')



#Testing:

with torch.no_grad():
    X_test = testx
    Y_test = testy

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: {}%'.format(accuracy.item()*100))
