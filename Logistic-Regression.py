#!/usr/bin/env python
# coding: utf-8

# In[119]:


# Author: Blake Edwards
import numpy as np
import torch
sigmoid = torch.nn.Sigmoid()

X = torch.Tensor([0.5,0.75,1.00,1.25,1.5,1.75,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,4.0,4.25,4.5,4.75,5.0,5.5])
Y = torch.Tensor([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1])

params = torch.randn(2).requires_grad_()
lr = 0.3

# operations
def f(x, params):
    w, b = params
    z = w*x+b
    preds = torch.exp(z)/(1+torch.exp(z))
    return preds
    
def cost_mle(y, y_hat):
    return -1*torch.mean(y*torch.log(y_hat) + (1-y)*torch.log(1-y_hat))

def cost_map(y, y_hat):
    w,b = params.data
    return -1*(torch.mean(y*torch.log(y_hat) + (1-y)*torch.log(1-y_hat)) + (w*w))

def cost_mse(y, y_hat):
    return (1/2)*torch.mean((y_hat-y)*(y_hat-y))


def f_classify(x, params):
    preds = f(x, params)
    classification_preds = torch.round(preds)
    return classification_preds
    
def calc_accuracy(x,y,params):
    classification_preds = f_classify(x,params)
    acc = 1-torch.mean(torch.abs(Y-classification_preds))
    return acc

cost = cost_map
# cost = cost_mle

print("---------------Start----------------")
print(f"params: {params.data}")
preds = f(X,params)
loss = cost(Y,preds)
acc = calc_accuracy(X,Y,params)
print(f"predictions: {preds}")
print(f"accuracy: {acc}")
print(f"loss: {loss}")
    
    
for i in range(2000):
    preds = f(X,params)
    loss = cost(Y,preds)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if i !=0 and i%500 == 0:
        print(f"\n---------------Epoch {i}----------------")
        preds = f(X,params)
        loss = cost(Y,preds)
        acc = calc_accuracy(X,Y,params)
        print(f"params: {params.data}")
        print(f"accuracy: {acc}")
        print(f"loss: {loss}")


print("\n---------------End----------------")
print(f"params: {params.data}")
preds = f(X,params)
loss = cost(Y,preds)
acc = calc_accuracy(X,Y,params)
print(f"predictions: {preds}")
print(f"accuracy: {acc}")
print(f"loss: {loss}")





