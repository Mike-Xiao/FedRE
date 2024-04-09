
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def compute_sens(model: nn.Module,
                 rootset_loader: DataLoader, 
                 device: torch.device,
                 loss_fn = nn.CrossEntropyLoss()):
    model = model.to(device)
    loss_fn = loss_fn.to(device).requires_grad_()
    x, y = next(iter(rootset_loader))
    x = x.to(device).requires_grad_()
    y = y.to(device).requires_grad_()

    # Compute prediction and loss
    pred = model(x).requires_grad_()
    print("pred shape is",pred.shape)
    print("pred shape is", pred.shape)
    print("Max value in pred is", pred.max().item())
    print("Min value in pred is", pred.min().item())
    print("y shape is",y.shape)
    print("pred is",pred)
    
    pred = pred.squeeze(1)
    y = y.long()
    loss = loss_fn(pred, y).requires_grad_()
    print("loss is",loss)
    
    jacobian = torch.autograd.grad(outputs=pred, 
                               inputs=x, 
                               grad_outputs=torch.ones_like(pred),
                               create_graph=True)[0]
    
    
    
    
    # Backward propagation
    dy_dx = torch.autograd.grad(outputs=loss, 
                                inputs=model.parameters(),
                                create_graph=True,
                                allow_unused=True)
    
    
    for name, param in model.named_parameters():
        print(f"Gradient of parameter {name}: {param.grad}")
    # for name, param in model.original_model.named_parameters():
    #         if param.grad_fn is None:
    #             print(f"Parameter {name} was not used in the computation.")
    print("dy_dx:",dy_dx)
    # for name, grad in zip(model.named_parameters(), dy_dx):
    #     if grad is None or torch.all(grad == 0):
    #         print(f"Gradient of {name} is zero.")
    print("dy_dx.shape is",len(dy_dx))
    vector_jacobian_products = []
    
    
    
    
    for layer in dy_dx:
        # Input-gradient Jacobian
        d2y_dx2 = torch.autograd.grad(outputs=layer, 
                                      inputs=x, 
                                      grad_outputs=torch.ones_like(layer),
                                      retain_graph=True)[0]
        vector_jacobian_products.append(d2y_dx2.detach().clone())

    vector_jacobian_products_cpu = [tensor.cpu() for tensor in vector_jacobian_products]



    sensitivity = []
    for layer_vjp in vector_jacobian_products:
        f_norm_sum = 0
        for sample_vjp in layer_vjp:
            # Sample-wise Frobenius norm
            f_norm_sum += torch.norm(sample_vjp)
        f_norm = f_norm_sum / len(layer_vjp)
        sensitivity.append(f_norm.cpu().numpy())
    
    return sensitivity, vector_jacobian_products_cpu



