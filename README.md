# smartalec
PyTorch implementation of linear equality constraints in feedforward neural networks

This is an implementation of the Lagrange parameter-based constraint satisfaction neural network described in https://arxiv.org/abs/2211.04395. It is completely self contained and does not require any datasets. Essentially, there is one python file which when executed produces the results in the referenced arXiv manuscript.

Before entering into the description of the code and how it works, we list the current dependencies of the program. These were obtained by executing pipreqs on the directory containing the code. Please see requirements.txt for the actual pipreqs output.

Requirements:

matplotlib==3.6.2

numpy==1.21.5

torch==1.12.0

Structure of the code:

Brief summary: The code builds a fairly standard XOR MLP with 4 fully connected layers. The last layer (fc4) is a linear classifier with a single output. The fc3 layer weights are the focus of this code. These weights are stored as the A matrix (referred to in the linked manuscript). Then we construct XOR Lagrange with an fc1 layer and a special Lagrange layer which comprises fully connected weights W (with bias) and the stored fixed weights A (with no bias). In addition to the standard forward function, Lagrange includes solver and line_search with the former executing gradient descent w.r.t. the Lagrange parameters (one set per pattern) and the latter computing a suitable step size value for the parameter alpha. Constrained_XOR builds the new XOR Lagrange model. Training the new model is accomplished in XOR_CS_Train.

Notes: 

1. A is defined using register_buffer and is therefore never updated. This is explicitly checked at the end.
2. line_search is a recursive function which solves for a suitable value of alpha.
3. We use forward hooks to obtain (and store) intermediate layer values (in XOR_layer_outputs). This is needed to compute A*y.
4. We set up a specialized loss function Lagrange_Loss but in this version is basically BCEWithLogitsLoss.
5. mutrain_loader contains the Lagrange parameters (and we use mu instead of lambda as in the manuscript for obvious reasons). 
6. mutrain contains the historical record (the state) of the Lagrange parameters and we use PyTorch's detach to ensure that gradient computation doesn't go off the rails. This is very important.  
7. XOR_CS_Plots is responsible for generating all the plots reported in the manuscript.

Inputs: None. The XOR data are generated in XOR_data.

Outputs: The loss function, the norm of the Lagrange parameters and the number of constraint satisfaction iterations in each epoch and in each run are stored in a pickle file. 

Free parameters: All noted in main.

Plots: The plots are generated in a CS_iter_NN_loss_Lag_param....... file. An example (for number_runs = 10) has been uploaded.

