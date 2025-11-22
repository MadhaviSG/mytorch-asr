import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)

        exp_Z = np.exp(Z_shifted)

        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        raise self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
        
        # Store original shape and A
        original_A = self.A
        original_dLdA = dLdA

        # Reshape input to 2D
        if len(shape) > 2:
            A_moved = np.moveaxis(self.A, self.dim, -1)
            dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
            
            # Flatten all other dimensions
            batch_size = np.prod(A_moved.shape[:-1])
            self.A = A_moved.reshape(batch_size, C)
            dLdA = dLdA_moved.reshape(batch_size, C)

        batch_size = self.A.shape[0]
        dLdZ = np.zeros_like(self.A)
        
        for i in range(batch_size):
            # Get the softmax output for this sample
            a = self.A[i:i+1, :]  # Shape: (1, C)
            # Compute Jacobian: J[m,n] = a[m] * (δ[m,n] - a[n])
            # This can be written as: J = diag(a) - a^T @ a
            jacobian = np.diag(a.flatten()) - a.T @ a
            # Compute gradient: dL/dZ = dL/dA @ J
            dLdZ[i:i+1, :] = dLdA[i:i+1, :] @ jacobian

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            dLdZ = dLdZ.reshape(A_moved.shape)
            # Move dimension back to original position
            dLdZ = np.moveaxis(dLdZ, -1, self.dim)
            # Restore original A
            self.A = original_A

        return dLdZ