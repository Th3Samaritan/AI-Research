"""
Chapter 2: Automatic Differentiation Engine
Mini forward-mode and reverse-mode AD engine.
"""
import numpy as np

class Dual:
    """Forward-mode AD scalar dual number."""
    def __init__(self, val, grad=0.0):
        self.val = val
        self.grad = grad

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val + other.val, self.grad + other.grad)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val * other.val, self.val * other.grad + self.grad * other.val)

    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __sub__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val - other.val, self.grad - other.grad)

    def __pow__(self, power):
        return Dual(self.val ** power, power * (self.val ** (power - 1)) * self.grad)
        
    def __repr__(self):
        return f"Dual(val={self.val}, grad={self.grad})"

def sin(x):
    if isinstance(x, Dual):
        return Dual(np.sin(x.val), np.cos(x.val) * x.grad)
    elif isinstance(x, Node):
        out = Node(np.sin(x.val), (x,), 'sin')
        def _backward():
            x.grad += np.cos(x.val) * out.grad
        out._backward = _backward
        return out
    return np.sin(x)

class Node:
    """Reverse-mode AD computational graph node."""
    def __init__(self, val, children=(), op=''):
        self.val = val
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.val + other.val, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
        
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.val * other.val, (self, other), '*')

        def _backward():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Node(self.val ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.val**(other-1)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Node(val={self.val}, grad={self.grad})"

if __name__ == "__main__":
    print("--- Forward Mode AD ---")
    # f(x, y) = x^2 * y + sin(x)
    # df/dx = 2xy + cos(x)
    x = Dual(2.0, grad=1.0) # We want derivative w.r.t x
    y = Dual(3.0, grad=0.0)
    
    z = (x ** 2) * y + sin(x)
    print("Function value:", z.val)
    print("Derivative w.r.t x:", z.grad)
    print("Expected df/dx at (2,3):", 2*2*3 + np.cos(2))
    print()
    
    print("--- Reverse Mode AD ---")
    x_node = Node(2.0)
    y_node = Node(3.0)
    
    z_node = (x_node ** 2) * y_node + sin(x_node)
    z_node.backward()
    
    print("Function value:", z_node.val)
    print("Derivative w.r.t x:", x_node.grad)
    print("Derivative w.r.t y:", y_node.grad)
    print("Expected df/dy at (2,3):", 2**2)
