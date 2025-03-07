 # 1) Design model (input, output size, forward pass)
 # 2) Construct loss and iptimizer
 # 3) Training loop
 # - forward pass: compute prediction
 # - backward pass: gradient
 # - update weights 

import torch 
import torch.nn as nn 

# f = w * x 

# f = 2 * x

X = torch.tensor([[1] ,[2] , [3] , [4]] , dtype=torch.float32)
Y = torch.tensor([[2] ,[4] , [6] , [8]] , dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_fearures = X.shape
print(n_samples, n_fearures )

input_size = n_fearures
output_size = n_fearures
	
#model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module): 
    
    def __init__(self, input__dim, output__dim):
        super(LinearRegression, self).__init__()
        #define our layer
        self.lin = nn.Linear(input__dim, output__dim)

        
    def forward(self, x): 
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#Training
learning_rate = 0.01
n_iters = 100 

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters): 
    # prediction = forwardpass
    y_pred = model(X)
    
    #loss
    l = loss(Y,  y_pred)
    
    #gradients = backward pass
    l.backward()  #dl/dw
    
    #update weight
    optimizer.step()
    
    #rero gradient
    optimizer.zero_grad()
    
    if epoch % 2 == 0: 
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}:w = {w[0][0].item():3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')     