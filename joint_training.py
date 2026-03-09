from src.utils.datasets import CustomImageDataset

import torch 
from torch.utils.data import DataLoader
import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

train = CustomImageDataset(csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/batches/batch-1024.csv", autoencoder=False)
valid = CustomImageDataset(csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/PUC_valid.csv", autoencoder=False)
test = CustomImageDataset(csv="/home/lucas.ocunha/ConditionalAutoencoder/CSV/PUC/PUC_test.csv", autoencoder=False)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid, batch_size=32, shuffle=False)
test_loader = DataLoader(test, batch_size=32, shuffle=False)


 
# Define models
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 32 * 32, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc1(x)
        return x
 
 
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(10, 20, 1, batch_first=True)
        self.fc = nn.Linear(20, 10)
 
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
 
 
cnn = SimpleCNN()
rnn = SimpleRNN()
 
# Initialize parameters
import torch.nn.init as init
 
for m in cnn.modules():
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
 
for m in rnn.modules():
    if isinstance(m, nn.RNN):
        init.orthogonal_(m.weight_hh_l0)
        init.xavier_normal_(m.weight_ih_l0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
 
# Define loss function
criterion_cnn = nn.CrossEntropyLoss()
criterion_rnn = nn.CrossEntropyLoss()
 
 
def combined_loss(output_cnn, output_rnn, target):
    loss_cnn = criterion_cnn(output_cnn, target)
    loss_rnn = criterion_rnn(output_rnn, target)
    return loss_cnn + loss_rnn
 
 
# Define optimizer
parameters = list(cnn.parameters()) + list(rnn.parameters())
optimizer = optim.Adam(parameters, lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
 
# TensorBoard writer
writer = SummaryWriter()
 
# Training data
input_cnn = torch.randn(10, 3, 32, 32)
input_rnn = torch.randn(10, 5, 10)
target = torch.randint(0, 10, (10,))
 
num_epochs = 10
best_val_loss = float('inf')
patience = 5
counter = 0
for epoch in range(num_epochs):
    optimizer.zero_grad()
 
    output_cnn = cnn(input_cnn)
    output_rnn = rnn(input_rnn)
 
    loss = combined_loss(output_cnn, output_rnn, target)
    loss.backward()
 
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
 
    optimizer.step()
    scheduler.step()
 
    writer.add_scalar('Loss/train', loss.item(), epoch)
 
    # Simulate validation loss
    val_loss = loss.item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            break
 
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
 
writer.close()