"""
Chapter 8: NumPy Internals, Strides, Broadcasting, and Vectorization
"""
import numpy as np
import timeit

def demo_strides():
    print("--- Memory Layout and Strides ---")
    arr = np.arange(10, dtype=np.int32).reshape(2, 5)
    print(f"Array shape: {arr.shape}")
    print(f"Array strides: {arr.strides} (bytes to step in each dimension)")
    
    # Transpose changes strides, not memory
    arr_t = arr.T
    print(f"Transposed shape: {arr_t.shape}")
    print(f"Transposed strides: {arr_t.strides}")
    print(f"Shares memory: {np.shares_memory(arr, arr_t)}\n")

def demo_broadcasting():
    print("--- Broadcasting ---")
    x = np.array([1, 2, 3])
    y = np.array([[10], [20], [30]])
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    
    z = x + y
    print("Result of x + y:")
    print(z)
    print(f"Result shape: {z.shape}\n")

def python_loop(arr1, arr2):
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        result[i] = arr1[i] + arr2[i]
    return result

def vectorized_op(arr1, arr2):
    return arr1 + arr2

def demo_vectorization():
    print("--- Vectorization Speedup ---")
    N = 1000000
    arr1 = np.random.rand(N)
    arr2 = np.random.rand(N)
    
    # Benchmark loop
    t_loop = timeit.timeit(lambda: python_loop(arr1, arr2), number=10)
    print(f"Python loop time (10 runs): {t_loop:.4f} seconds")
    
    # Benchmark vectorized
    t_vec = timeit.timeit(lambda: vectorized_op(arr1, arr2), number=10)
    print(f"Vectorized time (10 runs): {t_vec:.4f} seconds")
    
    speedup = t_loop / t_vec
    print(f"Speedup: {speedup:.2f}x\n")

if __name__ == "__main__":
    demo_strides()
    demo_broadcasting()
    demo_vectorization()
