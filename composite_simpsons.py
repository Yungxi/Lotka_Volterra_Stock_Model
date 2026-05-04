"""
Composite Simpson's Rule for Numerical Integration
"""


def composite_simpsons(f, a, b, n):
    """
    Approximate the definite integral of f(x) from a to b using
    the composite Simpson's rule.

    Parameters:
        f: Function to integrate (callable)
        a: Lower bound of integration
        b: Upper bound of integration
        n: Number of subintervals (must be even)

    Returns:
        Approximation of the integral

    Raises:
        ValueError: If n is not even or n < 2
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")

    h = (b - a) / n

    # First and last terms
    result = f(a) + f(b)

    # Sum odd indices (coefficient 4)
    for i in range(1, n, 2):
        result += 4 * f(a + i * h)

    # Sum even indices (coefficient 2)
    for i in range(2, n, 2):
        result += 2 * f(a + i * h)

    return result * h / 3


if __name__ == "__main__":
    import math

    # ============ EDIT INPUTS HERE ============
    def f(x):
        return math.tan(x)  # Change this function

    a = 0          # Lower bound
    b = (math.pi/4)    # Upper bound
    n = 6        # Number of subintervals (must be even)
    # ==========================================

    result = composite_simpsons(f, a, b, n)
    print(f"∫ f(x) dx from {a} to {b}")
    print(f"n = {n} subintervals")
    print(f"Result ≈ {result:.10f}")
