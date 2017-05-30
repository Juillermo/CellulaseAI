# CellulaseAI
Shallow MLP with back-propagation for making feeding decision in a fermentation process based on sensor signals.

Shallow feed-forward multi-layer perceptron trained with back-propagation, batch training, competitive learning and L2 regularization. The MSE is the cost function and sigmoids are used as activation function
The inputs are analog data from three sensors (pH, DO, and ORP) which constantly measure the fermentation conditions and some variations of them with exponential smoothing and differential filters.
