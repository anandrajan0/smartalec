# smartalec
PyTorch implementation of linear equality constraints in feedforward neural networks

This is an implementation of the Lagrange parameter-based constraint satisfaction neural network described in https://arxiv.org/abs/2211.04395. It is completely self contained and does not require any datasets. Essentially, there is one python file which when executed produces the results in the referenced arXiv manuscript.

Before entering into the description of the code and how it works, we list the current dependencies of the program. These were obtained by executing pipreqs on the directory containing the code. Please see requirements.txt for the actual pipreqs output.

Requirements:

matplotlib==3.6.2

numpy==1.21.5

torch==1.12.0
