"""
Chapter 9: Custom torch.autograd.Function
NOTE: This script simulates wrapping a custom CUDA operation for PyTorch.
It requires PyTorch, and normally requires an NVIDIA GPU and custom CUDA C++ extensions.
Here we implement a Python fallback for demonstration of autograd mechanics.
"""
import torch

class CustomLaplacian(torch.autograd.Function):
    """
    A custom autograd function that applies a 1D Laplacian (finite difference).
    In a real scenario, the forward/backward passes could call custom C++/CUDA kernels.
    """
    @staticmethod
    def forward(ctx, input_tensor, dx):
        # Save variables for the backward pass
        ctx.dx = dx
        
        # Output tensor of the same shape
        output = torch.zeros_like(input_tensor)
        
        # Compute finite difference Laplacian (central difference)
        # Using pure PyTorch operations here; in practice this would be a custom kernel launch.
        output[1:-1] = (input_tensor[2:] - 2 * input_tensor[1:-1] + input_tensor[:-2]) / (dx**2)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        dx = ctx.dx
        
        # We need to compute the gradient with respect to the input_tensor.
        # The operation is a linear convolution, so the adjoint (backward) is also a convolution.
        grad_input = torch.zeros_like(grad_output)
        
        # Backward finite difference (adjoint of the forward Laplacian)
        grad_input[1:-1] += -2 * grad_output[1:-1] / (dx**2)
        grad_input[2:] += grad_output[1:-1] / (dx**2)
        grad_input[:-2] += grad_output[1:-1] / (dx**2)
        
        # Gradient w.r.t dx is not required (return None)
        return grad_input, None

def custom_laplacian(input_tensor, dx):
    return CustomLaplacian.apply(input_tensor, dx)

def main():
    print("--- PyTorch Custom Autograd Function Demo ---")
    # Requires NVIDIA GPU for real CUDA, but we run on CPU/CUDA depending on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1D domain
    nx = 10
    dx = 0.1
    x = torch.linspace(0, 1, nx, requires_grad=True, device=device)
    
    # Apply our custom operation
    lap = custom_laplacian(x, dx)
    
    print("Input x:")
    print(x)
    print("\nLaplacian output:")
    print(lap)
    
    # Define a scalar loss
    loss = lap.sum()
    print(f"\nLoss: {loss.item()}")
    
    # Backward pass
    loss.backward()
    
    print("\nGradient of x (x.grad):")
    print(x.grad)
    print("\nNotice that gradients flowed through our CustomLaplacian!")

if __name__ == "__main__":
    main()
