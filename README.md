# smartalec
PyTorch implementation of linear equality constraints in feedforward neural networks

This is an implementation of the Lagrange parameter-based constraint satisfaction neural network described in https://arxiv.org/abs/2211.04395. It is completely self contained and does not require any datasets. Essentially, there is one python file which when executed produces the results in the referenced arXiv manuscript.

Before entering into the description of the code and how it works, we list the current dependencies of the program. These were obtained by executing pipreqs on the directory containing the code. Please see requirements.txt for the actual pipreqs output.

Requirements:

matplotlib==3.6.2

numpy==1.21.5

torch==1.12.0

Structure of the code:

Brief summary: The code builds a fairly standard XOR MLP with 4 fully connected layers. The last layer (fc4) is a linear classifier with a single output. The fc3 layer weights are the focus of this code. These weights are stored as the A matrix (referred to in the linked manuscript). Then we construct XOR Lagrange with an fc1 layer and a special Lagrange layer which comprises standard fully connected weights W (with bias) and the stored fixed weights A (with no bias). In addition to the standard forward function, Lagrange includes solver and line_search with the former executing gradient descent w.r.t. the Lagrange parameters (one set per pattern) and the latter computing a suitable step size parameter.
