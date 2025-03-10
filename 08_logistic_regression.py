import torch  
import torch.nn as nn   
import numpy as np   
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
 
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Scale 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

# 1) Model: Logistic Regression (f = wx + b, with sigmoid)
class LogisticRegression(nn.Module): 
    def __init__(self, n_input_features): 
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
     
    def forward(self, x): 
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features)
  
# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
 
# 3) Training loop 
num_epochs = 100
for epoch in range(num_epochs): 
    # Forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0: 
        print(f"Epoch: {epoch+1}, Loss = {loss.item():.4f}")
        
# Evaluation
with torch.no_grad(): 
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = (y_predicted_cls == y_test).float().mean()
    print(f'Accuracy = {acc:.4f}')
