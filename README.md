# Sidechainnet Atomic Structure Minimizer using Energy-Force model. 

The structure minimizer uses a generic pytorch layer that calculates potential energy in forward pass and returns force gradients per atom in backward pass. The pytorch layer is an extension of pytorch autograd mechanics.
