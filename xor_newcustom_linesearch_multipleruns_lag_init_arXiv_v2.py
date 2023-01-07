#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 5 20:14:12 2022

@author: Anand Rangarajan, Pan He 
"""

# The XOR problem and constraints
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


# Creation of datasets
def XOR_data():
    
    N2 = N - N1
    N11 = int(N1/2)
    N12 = N1 - N11
    N21 = int(N2/2)
    N22 = N2 - N21
    
    # Values of spread less than 1 cause pattern overlap
    spread = 1.0    # Too trivial to cast as nonlocal. Set once and never changed.
    
    random_seed = random.randint(0, 1e+8)
    torch.manual_seed(random_seed)
    x = 2*np.random.rand(N, D) - 1
    x[0:N11,:] += np.outer(np.ones(N11),np.array([1,1])*spread)
    x[N11:N1,:] += np.outer(np.ones(N12),np.array([-1,-1])*spread)
    x[N1:N1+N21,:] += np.outer(np.ones(N21),np.array([-1,1])*spread)
    x[N1+N21:N,:] += np.outer(np.ones(N22),np.array([1,-1])*spread)
    y = np.zeros(N)
    y[0:N1] = np.ones(N1)
    y[N1:N] = np.zeros(N2)

    # Adding labels and an index to x
    xy = np.concatenate((x, 
         np.concatenate((np.array([y]), np.array([np.arange(0,N)]))).T),\
                        axis=1)
    return xy

# The XOR MLP without constraint satisfaction
class XOR(nn.Module):   
    def __init__(self, input_size, hidden_size, constraint_size, output_size):
        super().__init__()
        random_seed = random.randint(0,1e+8)
        torch.manual_seed(random_seed)
        
        # 2x5
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),     
            nn.Sigmoid()
        )
        torch.nn.init.normal_(self.fc1[0].weight, mean=0.0, std=1)
        torch.nn.init.zeros_(self.fc1[0].bias)
        # 5x10
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=output_size),           
            nn.Sigmoid()
        )   
        torch.nn.init.normal_(self.fc2[0].weight, mean=0.0, std=1)
        torch.nn.init.zeros_(self.fc2[0].bias)
        # 10x4
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=output_size, out_features=constraint_size),
            nn.Sigmoid()
        )
        torch.nn.init.normal_(self.fc3[0].weight, mean=0.0, std=1)
        torch.nn.init.zeros_(self.fc3[0].bias)
        # 4x1
        # Note linear output layer. Examine the loss function (BCEWithLogitsLoss)
        self.fc4 = nn.Linear(in_features=constraint_size, out_features=1)
        torch.nn.init.normal_(self.fc4.weight, mean=0.0, std=1)
        torch.nn.init.zeros_(self.fc4.bias)
        
        # Creating an activation directory with forward hooks
        self.activation = {}
        
        
    def forward(self, x):
        z = self.fc1(x)
        z = self.fc2(z)
        z = self.fc3(z)
        z = self.fc4(z)
        z = torch.squeeze(z)
        return z
    
    
    # Defining a forward hook for activations
    # This is poorly documented but is activated when forward is executed
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

# Training the XOR model    
def XOR_train_test(device, train_data, test_data):
          
    # setting up the XOR model
    model = XOR(input_size, hidden_size, constraint_size, output_size)
    model = model.float()
    model.to(device)
     
    error = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_XOR)
    print(model)
       
    accuracy_hist = np.zeros(num_epochs_XOR)
    accuracy_test_hist = np.zeros(num_epochs_XOR)
    
    print("XOR training")
    XOR_loss_loader = torch.zeros([num_epochs_XOR, batches], device = "cuda:0")
    for epoch in range(num_epochs_XOR):
        # Randomly permuting the data in each epoch
        train_data = train_data[torch.randperm(N)]
        for index in range(batches):
            patterns = train_data[index*batch_size:(index+1)*batch_size, 0:D]
            labels = train_data[index*batch_size:(index+1)*batch_size, D]
            
            # Forward pass 
            outputs = model(patterns)
            loss = error(outputs, labels)
            
            # Initializing a gradient as 0 so there is no mixing of gradients 
            # among the batches
            optimizer.zero_grad()
            
            #Propagating the error backward
            loss.backward()
            
            # Optimizing the parameters
            optimizer.step()
            
            is_correct = torch.round(torch.sigmoid(outputs)) == labels
            accuracy_hist[epoch] += is_correct.sum()/N
      
        for index in range(batches):
            patterns = test_data[index*batch_size:(index+1)*batch_size, 0:D]
            labels = test_data[index*batch_size:(index+1)*batch_size, D]
            outputs = model(patterns)
            is_correct = torch.round(torch.sigmoid(outputs)) == labels
            accuracy_test_hist[epoch] += is_correct.sum()/N
            
            XOR_loss_loader[epoch, index] = loss.data
            
        print("Run: {}, Epoch: {}, Loss: {:.5f}, Training Accuracy: {:.3f}, Testing Accuracy: {:.3f}".\
              format(runs, epoch, torch.mean(XOR_loss_loader[epoch, 0:batches]), \
                     accuracy_hist[epoch], accuracy_test_hist[epoch]))
    
    return model


# Code to produce intermediate outputs
def XOR_layer_outputs(model, test_data, name, size):
    # Saving the fc2 layer outputs of the test data   
    outputs_test_loader = torch.zeros([N, size], device = "cuda:0")
    for index in range(batches):
        patterns = test_data[index*batch_size:(index+1)*batch_size, 0:D]
        # Executing the model on a batch so that activations are populated
        model(patterns)
        # Getting the intermediate output of the batch
        outputs_test_loader[index*batch_size:(index+1)*batch_size,\
                                0:size] = model.activation[name]
    return outputs_test_loader



    
# The new XOR_CS lower layers without Lagrange
class XOR_LowerLayers(nn.Module):   
    def __init__(self, input_size, hidden_size):
        super().__init__()
        random_seed = random.randint(0,1e+8)
        torch.manual_seed(random_seed)
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        torch.nn.init.normal_(self.fc1[0].weight, mean=0.0, std=1)
        torch.nn.init.zeros_(self.fc1[0].bias)                
    
    def forward(self, x):
        z = self.fc1(x)
        return z
    

# The Lagrange class which includes constraint satisfaction    
class Lagrange(nn.Module):
    def __init__(self, hidden_size, constraint_size, output_size, constraint_matrix):
        super().__init__()
        random_seed = random.randint(0, 1e+8)
        torch.manual_seed(random_seed)
        # Defining the weights and bias of the last layer
        W = torch.randn(hidden_size, output_size, device = 'cuda:0')
        self.W = nn.Parameter(W)  # a Tensor that's a module parameter.
        bias = torch.randn(output_size, device = 'cuda:0')
        self.bias = nn.Parameter(bias)
        
        A_parameter = torch.tensor([constraint_size, output_size], device = 'cuda:0') 
        # Setting the fixed constraint weights to that of the learned fc3 layer
        A_parameter = constraint_matrix
        # "A" is set to the previously trained fc3 weights.
        # Defining a buffer containing A which will never be updated
        self.register_buffer("A", A_parameter)
        
    # Note that Lagrange.forward requires the Lagrange parameters mu as input
    def forward(self, x, mu):
        output = torch.add(torch.add(torch.mm(x, self.W), self.bias), 
                 torch.mm(mu, self.A))
        return output
    
    # The all important constraint satisfaction solver
    def solver(self, x, y, mu):      
        AT = torch.transpose(self.A, 0, 1)
        
        
        ##################################################################
        ##################################################################
        # alpha_fixed = 0.1   # Initial condition for the stepsize parameter
        ##################################################################
        ##################################################################
        
              
        # Initializing the Lagrange parameters (called mu not to be confused with python lambda)
        #mu = torch.zeros(batch_size, constraint_size, device = "cuda:0")
        
        # PyTorch recommends never running the forward function 
        # But I think this use case is fine.
        u = self.forward(x, mu)
        z = nn.Sigmoid()(u)
        # The all important gradient w.r.t. mu
        # Note that we actually impose Az=Ay and not Az=b
        mu_step = torch.mm(z-y, AT) 
          
        CS_iterations = 0
        while (torch.linalg.vector_norm(mu_step)/batch_size >= CS_threshold) \
            and (CS_iterations < CS_iter_maximum):
        
            loss_zero = MyLoss(u, y)
            alpha = alpha_fixed
            mu = self.line_search(u, y, mu, mu_step, loss_zero, alpha)
            
            u = self.forward(x, mu)
            z = nn.Sigmoid()(u)
            
            mu_step = torch.mm(z-y, AT)
             
            # Number of constraint satisfaction iterations
            CS_iterations += 1  
            
        CS_error = torch.linalg.vector_norm(mu_step)/batch_size
        return mu, CS_error, CS_iterations
    
    # Recursive function call to solve for alpha
    def line_search(self, u, y, mu, mu_step, loss_zero, alpha):
        
        #####################################################
        #####################################################
        # We stop the search when alpha drops below alpha_min
        # alpha_min = 1.0e-06
        #####################################################
        #####################################################
        
        u_alpha = torch.add(u,torch.mm(-alpha*mu_step, self.A))
        loss_alpha = MyLoss(u_alpha, y)       
        if (loss_alpha > loss_zero) & (alpha > alpha_min):
            alpha = alpha/2
            # Note: recursive call below
            return self.line_search(u, y, mu, mu_step, loss_zero, alpha)    
        return mu - alpha*mu_step

# The complete XOR_CS MLP comprising XOR_LowerLayers and Lagrange
class Constrained_XOR(nn.Module):   
    def __init__(self, input_size, hidden_size, constraint_size, output_size, constraint_matrix):       
        super().__init__()        
        self.XOR_LowerLayers = XOR_LowerLayers(input_size, hidden_size)
        self.Lagrange = Lagrange(hidden_size, constraint_size, output_size, constraint_matrix)     
    
    def forward(self, x, mu):
        z = self.XOR_LowerLayers(x)
        u = self.Lagrange(z, mu)
        return u

# The Lagrange loss function (basically BCEWithLogitsLoss)
class Lagrange_Loss(nn.Module):   
    def __init__(self):
        super().__init__()
    
    # This is basically BCEWithLogitsLoss
    # We wanted to add the y log(y) terms to make the loss lower bounded by 0
    def forward(self, outputs, targets):   
        return torch.mean(torch.add(-targets*outputs,torch.log(1+torch.exp(outputs))))+\
               torch.mean(torch.add(targets*torch.log(targets),\
                             (1-targets)*torch.log(1-targets)))
                   

# Training XOR_CS with history
def XOR_CS_train(device, test_data, outputs_fc2_test_loader, model_XOR):
    
    # Defining the constraint satisfaction model with Lagrange parameters
    model_CS = Constrained_XOR(input_size, hidden_size, \
        constraint_size, output_size, model_XOR.fc3[0].weight)
    model_CS.to(device)
      
    # Training parameters
    optimizer = torch.optim.Adam(model_CS.parameters(), lr=learning_rate_XOR_CS) 
     
    ###################################
    print("\nNAMED PARAMS")
    for name, param in model_CS.named_parameters():
        print("    ", name, "[", type(name), "]", type(param), param.size())
   
    print("\nNAMED BUFFERS")
    for name, buffs in model_CS.named_buffers():
        print("    ", name, "[", type(name), "]", type(buffs), buffs.size())
    print("\n Constraint Satisfaction epochs")
    print("----------------------------------")  
    
    # Main epoch iteration loop
    mutrain_loader = torch.zeros([batches*batch_size, constraint_size], device = "cuda:0")
    for epoch in range(num_epochs_XOR_CS):
        # Permuting the training set and the fc2 outputs which are targets
        perm_totalindex = torch.randperm(N)
        test_data = test_data[perm_totalindex]
        outputs_fc2_test_loader = outputs_fc2_test_loader[perm_totalindex]
        mutrain_loader = mutrain_loader[perm_totalindex]
            
        for index in range(batches):
            patterns = test_data[index*batch_size:(index+1)*batch_size, 0:D]
            targets_CS = \
                outputs_fc2_test_loader[index*batch_size:(index+1)*batch_size, 0:output_size]           
            targets_CS = targets_CS.to(device)
            # Ensuring that computed variables like mu which are non-leaf nodes 
            # do not have gradients computed. Detach them from the graph
            mutrain = mutrain_loader[index*batch_size:(index+1)*batch_size, 0:constraint_size]
            # Defining mutrain_no_grad as the detached variable: Warning - memory leak otherwise
            mutrain_no_grad = mutrain.detach()
            
            # Forward pass 
            # Uncomment the torch.autograd line below for debugging
            #torch.autograd.set_detect_anomaly(True)
            # Have to execute lower layers to produce the input for the 
            # Lagrange solver
            outputs_LL = model_CS.XOR_LowerLayers(patterns)
            # The Lagrange solver takes the output of the lower layers as the input
            mutrain, CS_error, CS_iterations = \
                model_CS.Lagrange.solver(outputs_LL, targets_CS, mutrain_no_grad)
            mutrain_no_grad = mutrain.detach()
            
            outputs_CS = model_CS(patterns, mutrain_no_grad)      
            
            # The BCEWithLogits Loss works with the linear predictor
            loss = MyLoss(outputs_CS, targets_CS)
            
            # Passing through a sigmoid after computing the loss function
            outputs_CS = nn.Sigmoid()(outputs_CS)
            
            # Computing a standard norm on the error
            NN_error = torch.linalg.vector_norm(outputs_CS - targets_CS)/batch_size
            mutrain_norm = torch.linalg.vector_norm(mutrain_no_grad)/batch_size
            print("Run: {}, epoch: {}, index: {}, NN_error: {:.5f}, CS_error: {:.5f}, CS_iterations: {}, Lag_param_norm: {:.5f}".format\
                  (runs, epoch, index, NN_error, CS_error, CS_iterations, mutrain_norm))
            
            # Initializing a gradient as 0 so there is no mixing of gradients 
            # among the batches
            optimizer.zero_grad()
            
            # Propagating the error backward
            # We should not need retain_graph=True for this to work
            loss.backward()
            
            # Optimizing the parameters
            optimizer.step()
            
            # Placing the Lagrange parameters back into a loader 
            mutrain_loader[index*batch_size:(index+1)*batch_size, 0:constraint_size] = mutrain_no_grad
            CS_iterations_loader[runs, epoch, index] = CS_iterations
            NN_loss_loader[runs, epoch, index] = loss.data
            mutrain_norm_loader[runs, epoch, index] = mutrain_norm
        print("Run: {}, Epoch: {}, Loss: {:.7f}, Avg. iterations: {:.1f}, Lag_param norm: {:.5f}".format(runs, epoch, \
            torch.mean(NN_loss_loader[runs, epoch, 0:batches]), \
            torch.mean(CS_iterations_loader[runs, epoch, 0:batches]), \
            torch.mean(mutrain_norm_loader[runs, epoch, 0:batches])))
            
    # Final error evaluations
    A = torch.zeros([constraint_size, output_size], device = 'cuda:0')
    A = model_CS.Lagrange.A
    AT = torch.transpose(A, 0, 1)
    NN_error_loader = torch.zeros(batches, device = "cuda:0")
    CS_error_loader = torch.zeros(batches, device = "cuda:0")
    outputs_CS_loader = torch.zeros([batches, batch_size, output_size], device = "cuda:0")
    
    for index in range(batches):
       patterns = test_data[index*batch_size:(index+1)*batch_size, 0:D]
       targets_CS = outputs_fc2_test_loader[index*batch_size:(index+1)*batch_size, 0:output_size]
       mutrain = mutrain_loader[index*batch_size:(index+1)*batch_size, 0:constraint_size]
        
       outputs_LL = model_CS.XOR_LowerLayers(patterns)
       mutrain, CS_error, CS_iterations = model_CS.Lagrange.solver(outputs_LL, targets_CS, mutrain)
       mutrain_no_grad = mutrain.detach()
        
       outputs_CS = model_CS(patterns, mutrain_no_grad)
       # The loss needs to be computed before the sigmoid
       loss = MyLoss(outputs_CS, targets_CS).detach()
       # Taking the sigmoid after computing the loss
       outputs_CS = nn.Sigmoid()(outputs_CS).detach()
       # Computing a separate NN error for the sake of sanity
       NN_error = torch.linalg.vector_norm(outputs_CS - targets_CS)/batch_size             
       CS_error = torch.linalg.vector_norm(torch.mm(outputs_CS-targets_CS, AT))/batch_size
       mutrain_norm = torch.linalg.vector_norm(mutrain_no_grad)/batch_size
       
       print("Run: {}, index: {}, NN_error: {:.5f}, CS_error: {:.5f}, CS_iterations: {}, Lag_param norm: {:.5f}"\
              .format(runs, index, NN_error.data, CS_error.data, CS_iterations, mutrain_norm))
                
       outputs_CS_loader[index, 0:batch_size, 0:output_size] = outputs_CS
       mutrain_loader[index*batch_size:(index+1)*batch_size, 0:constraint_size] = mutrain_no_grad
    
       CS_iterations_loader[runs, num_epochs_XOR_CS, index] = CS_iterations
       NN_loss_loader[runs, num_epochs_XOR_CS, index] = loss.data
       mutrain_norm_loader[runs, num_epochs_XOR_CS, index] = mutrain_norm
       # Just recording the NN and CS errors into a loader. Not plotted for now.
       NN_error_loader[index] = NN_error.detach()
       CS_error_loader[index] = CS_error.detach()
       
    print("Run: {}, Epoch: {}, Loss: {:.7f}, Avg. iterations: {:.1f}, , Lag_param norm: {:.5f}".format(runs, num_epochs_XOR_CS, \
        torch.mean(NN_loss_loader[runs, num_epochs_XOR_CS, 0:batches]), \
        torch.mean(CS_iterations_loader[runs, num_epochs_XOR_CS, 0:batches]),
        torch.mean(mutrain_norm_loader[runs, epoch, 0:batches])))
    #del mutrain_loader
    return model_CS
 
    
# All CS plots
def XOR_CS_plots(CS_iterations_loader, NN_loss_loader, mutrain_norm_loader):
    
    CS_iter_index_max = \
        torch.argmax(torch.sum(CS_iterations_loader, dim=[1,2]))
        
    CS_iter_index_min = \
        torch.argmin(torch.sum(CS_iterations_loader, dim=[1,2]))
        
    CS_iterations_max = CS_iterations_loader[CS_iter_index_max].cpu().numpy()
    NN_loss_max = NN_loss_loader[CS_iter_index_max].detach().cpu().numpy()
    mutrain_norm_max = mutrain_norm_loader[CS_iter_index_max].detach().cpu().numpy()
    
    CS_iterations_min = CS_iterations_loader[CS_iter_index_min].cpu().numpy()
    NN_loss_min = NN_loss_loader[CS_iter_index_min].detach().cpu().numpy()
    mutrain_norm_min = mutrain_norm_loader[CS_iter_index_min].detach().cpu().numpy()
       
    fig, ((axes_11, axes_12, axes_13), (axes_21, axes_22, axes_23), (axes_31, axes_32, axes_33)) \
        = plt.subplots(3, 3, layout = "constrained")
    fig.suptitle("Average number of iterations, loss and Lag. param. norm per epoch")
    #fig.tight_layout()
    axes_11.plot(np.mean(CS_iterations_max, axis = 1))
    axes_11.set(xlabel = "Epoch", ylabel = "iterations")
    axes_11.set_title("Maximum")
    axes_12.plot(np.mean(CS_iterations_min, axis = 1))
    axes_12.set(xlabel = "Epoch", ylabel = "iterations")
    axes_12.set_title("Minimum")
    axes_13.plot(np.mean(CS_iterations_loader.detach().cpu().numpy(), axis = (0,2)))
    axes_13.set(xlabel = "Epoch", ylabel = "iterations")
    axes_13.set_title("Mean")
    axes_21.plot(np.mean(NN_loss_max, axis = 1))
    axes_21.set(xlabel = "Epoch", ylabel = "average loss")
    axes_22.plot(np.mean(NN_loss_min, axis = 1))
    axes_22.set(xlabel = "Epoch", ylabel = "average loss")
    axes_23.plot(np.mean(NN_loss_loader.detach().cpu().numpy(), axis = (0,2)))
    axes_23.set(xlabel = "Epoch", ylabel = "average loss")
    axes_31.plot(np.mean(mutrain_norm_max, axis = 1))
    axes_31.set(xlabel = "Epoch", ylabel = r"$||\mathbf{\lambda}||_{2}$")
    axes_32.plot(np.mean(mutrain_norm_min, axis = 1))
    axes_32.set(xlabel = "Epoch", ylabel = r"$||\mathbf{\lambda}||_{2}$")
    axes_33.plot(np.mean(mutrain_norm_loader.detach().cpu().numpy(), axis = (0,2)))
    axes_33.set(xlabel = "Epoch", ylabel = r"$||\mathbf{\lambda}||_{2}$")
    plt.savefig('CS_iter_NN_loss_Lag_param_perm.pdf')  
    
    # A dictionary to save all the final outputs
    results_dict = {}
    results_dict = {"NN_loss" : NN_loss_loader, \
                    "CS_iterations" : CS_iterations_loader, \
                    "Lag. param. norm" : mutrain_norm_loader}

    filename = "Lagrange_results_perm.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
        
# The Main Program
def main():

    ##Achtung, Achtung, Achtung################################################
    ##Global declarations######################################################
    ###########################################################################
    ###########################################################################
    global device
    global num_features, D, num_patterns, N, N1
    global input_size, hidden_size, constraint_size, output_size
    global num_epochs_XOR, learning_rate_XOR, batch_size, num_epochs_XOR_CS,\
                                learning_rate_XOR_CS, batches, number_runs
    global layer_name, alpha_fixed, alpha_min, CS_threshold, \
                                CS_iter_maximum, MyLoss, runs
    global CS_iterations_loader, NN_loss_loader, mutrain_norm_loader
    ###########################################################################
    ###########################################################################


# Checking to see if we have a GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    
#   Training and test data and CS stands for constraint satisfaction
    num_features = 2 # Number of input feature dimensions
    D = num_features   # Number of input feature dimensions
    num_patterns = 2000 # Number of training patterns: same in XOR and XOR CS
    N = num_patterns    # Number of training patterns: same in XOR and XOR CS
    N1 = int(N/2) # Number of patterns in class 1 
    
#   Model parameters

    input_size = 2
    hidden_size = 5
    output_size = 10
    constraint_size = 4
    
#   Setting up all parameters
    num_epochs_XOR = 125
    learning_rate_XOR = 0.001
    batch_size = 100    # Same in both XOR and XOR CS
    num_epochs_XOR_CS = 100   # The number of epochs in CS
    learning_rate_XOR_CS = 0.01 # The learning rate in CS
    batches = int(N/batch_size) # Number of batches same in both XOR and XOR CS
    number_runs = 2   # The number of runs 
    
#   Constraint satisfaction parameters
    layer_name = "fc2"    # The XOR layer for which we obtain intermediate activations
    alpha_fixed = 0.1       # The initial condition for alpha
    alpha_min = 1.0e-06     # The floor for alpha
    CS_threshold = 0.001    # The constraint_satisfaction threshold
    CS_iter_maximum = 1000  # Maximum number of constraint satisfaction iterations
    runs = 0                # The parameter indicating which run we are on (global parameter)

#   Defining a new loss function to be used in constraint satisfaction 
#   (basically BCEWithLogits)
    MyLoss = Lagrange_Loss()

#   Setting up outputs that will be recorded
    CS_iterations_loader = \
        torch.zeros([number_runs, num_epochs_XOR_CS+1, batches], device = "cuda:0")
    NN_loss_loader = \
        torch.zeros([number_runs, num_epochs_XOR_CS+1, batches], device = "cuda:0")
    mutrain_norm_loader = \
        torch.zeros([number_runs, num_epochs_XOR_CS+1, batches], device = "cuda:0")
    
#   Setting up multiple runs

    for runs in range(0, number_runs):
    
        train_data = XOR_data()
#       Random number generator for training and test sets
        random_seed = random.randint(0, 1e+8)
        torch.manual_seed(random_seed)
        train_data = torch.from_numpy(train_data)
        train_data = train_data.float()
        train_data = train_data.to(device)
    
        test_data = XOR_data()
        test_data = torch.from_numpy(test_data)  
        test_data = test_data.float()
        test_data = test_data.to(device)
    
    #   Building the XOR model
        model = XOR_train_test(device, train_data, test_data)
    #   Registering forward hooks to extract the fc2 outputs
        for name, layer in model.named_modules():
            layer.register_forward_hook(model.get_activation(name))
    #   Obtaining the fc2 layer outputs from the XOR model
    #   Note that test_data has not been permuted
        outputs_fc2_test_loader = XOR_layer_outputs(model, test_data, layer_name, output_size)
    
    #   Building XOR model_CS
    #   Warning: test_data gets permuted inside XOR_CS_train
        model_CS  = XOR_CS_train(device, test_data, outputs_fc2_test_loader, model)
    
    #   Printing all weights and biases
        print("The entire constraint satisfaction model")
        print("-------------------------------------------------------------")
        for param in model_CS.parameters():
            print(param.data)
        print("-------------------------------------------------------------")
    
        print("Original values of A")
        print(model.fc3[0].weight)
        print("-------------------------------------------------------------")
    
        print("Final values of A")
        print(model_CS.Lagrange.A)
        print("-------------------------------------------------------------")
    
    
    #   Deleting the models and important parameters
        model.cpu()
        del model
        del layer
        del outputs_fc2_test_loader
        model_CS.cpu()
        del model_CS
        del train_data
        del test_data
        torch.cuda.empty_cache()
      
#   Plot the results
    XOR_CS_plots(CS_iterations_loader, NN_loss_loader, mutrain_norm_loader)


# Run main function ######
if __name__ == "__main__":
    main()
    



    
    
    




