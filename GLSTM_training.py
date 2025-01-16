import torch
import torch.nn as nn

def train2(model, data, criterion, optimizer):
    model.train() # set model to training mode
    optimizer.zero_grad() # clear gradients 
    out = model(data.x, data.edge_index)#forward pass
    loss = criterion(out, data.y)#calculate loss
    loss.backward() #back propagation
    # Accumulate gradients for plotting
    gradients = []
    param_names = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append(param.grad.norm().item())  # Store gradient norm
            param_names.append(name)
    optimizer.step() #update weights
    pred = out.argmax(dim=1)
    correct = pred.eq(data.y).sum().item()
    accuracy = correct / data.num_nodes
    
    return loss.item(), accuracy, gradients, param_names

# Evaluation function
def evaluate(model, data, criterion, optimizer):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index) #forward pass
        loss = criterion(out, data.y) #calculate loss
        pred = out.argmax(dim=1) 
        correct = pred.eq(data.y).sum().item()
        accuracy = correct / data.num_nodes
    return loss.item(), accuracy

