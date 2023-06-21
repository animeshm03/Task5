### Task 5 (Machine Learning)

I have implemented a feedforward neural network from scratch. Have adhered to most of the guidelines provided in the doc, apart from a few things. A few details:

- `anneal`
    - Only support for fixed learning rate. Annealing not implemented.
- `opt`
    - support for only `gd` and `momentum`. `adam` and `nag` not implemented.
- `logs`
    - I have the logs on the training and the validation set in a single file. Different files not used for valid/train logs.