from operator import ixor
from functools import reduce
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from tqdm import tqdm

def make_binary_arrays(d=2):
    return np.array(
        list(
            map(lambda x: np.array(list("{0:b}".format(x).zfill(d))), 
                range(int(2**d))))
            ).astype(np.int64)




def batch_matvec(A,B):
    """i: sample
        j: input_dim
        k: hidden_dim


        [hidden_dim,input_dim]
        [sample_size, input_dim]

        [sample_size, hidden_dim]"""
    return torch.einsum('kj,ij->ik', A, B)



class Model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=3, out_dim=1, add_bias = False):
        super(Model, self).__init__()
        self.W1 = nn.Parameter(torch.rand(hidden_dim, input_dim))
        if add_bias:
            self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
            self.bias2 = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias1 = torch.zeros(hidden_dim)
            self.bias2 = torch.zeros(out_dim)
        self.W2 = nn.Parameter(torch.rand(out_dim, hidden_dim))
        
    def forward(self,inp):
        y1 = batch_matvec(self.W1, inp) + self.bias1
        y2 = torch.sigmoid(batch_matvec(self.W2, torch.sigmoid(y1)) + self.bias2) 
        return y1, y2.squeeze()
        

def run_optimization(model, B, target, n_iter=1000, lr=1e-1):
    opt = Adam(model.parameters(), lr=lr)
    losses = []
    for k in tqdm(range(n_iter)):
        y1, out = model(B)
        opt.zero_grad()
        sqr_error = (target-out)**2
        loss = torch.sum(sqr_error)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
                           
                           
def generalize_xor(B):           
    """
    (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 0

    (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)

    (0,0), (0,1), (0,1), (0,0), (1,0), (1,1), (1,1), (1,0)

    0, 1, 1, 0, 1, 0, 0, 1
    """
    o = np.empty([B.shape[0]], dtype = int)
    for i in range(B.shape[0]):
        xor = reduce(ixor, B[i, :])
        o[i] = xor
    return o
                          
def make_explicit_line_equation(w, b):
                           
    """   x w_0 + y w_1 = -b

    y = ..."""
                           
    w[0], w[1]
    def explicit_line_equation(x):
        "code for problem 1.3 here"
        y = (1 / w[1]) * (-b - w[0] * x)
        return y
        
    return explicit_line_equation


if __name__ == "__main__":
    
    B = make_binary_arrays(2)
    target = np.bitwise_xor(B[:,0],B[:,1])
                           
    #     Bnd = make_binary_arrays(3)
    #     targetnd = generalized_xor(Bnd)
    B = torch.from_numpy(B).float()
    target = torch.from_numpy(target).float()
    
    model = Model(B.shape[1], hidden_dim=2)
    #losses = run_optimization(model, B, target, error_fn=lambda x: skewed_error(x,alpha=0), n_iter=1000)
    losses = run_optimization(model, B, target, n_iter=1000)
    

    W1_np = model.W1.detach().numpy()
    W2_np = model.W2.detach().numpy()
    bias1_np = model.bias1.detach().numpy()
    bias2_np = model.bias2.detach().numpy()
    
    print("W1: {}".format(W1_np))
    print("W2: {}".format(W2_np))
    print("Bias 1: {}".format(bias1_np))
    print("Bias 2: {}".format(bias2_np))
