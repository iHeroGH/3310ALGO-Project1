"""
Implements methods for classic, d&q, and strassen matrix multiplication
"""
from typing import Callable

def get_width(matrix: list[list[int]]) -> int:
    """Returns the length of the longest sub-list in the matrix"""
    width = 0
    for i in matrix:
        if len(i) > width:
            width = len(i)
    return width

def can_multiply(
            matrix_a: list[list[int]],
            matrix_b: list[list[int]]
        ) -> bool:
    """Denotes whether or not two matrices can be multiplied"""
    return get_width(matrix_a) == len(matrix_b)

def matrix_multiplication(
            multiplier: Callable[
                            [list[list[int]], list[list[int]]],
                            list[list[int]]
                        ]
        ):
    """A decorator for a function taking two matrices and returning a matrix"""

    def wrapper(
                matrix_a: list[list[int]],
                matrix_b: list[list[int]]
            ):
        """
        Procedure to take place any time a matrix multiplication function
        is called
        """

        assert can_multiply(matrix_a, matrix_b)
        matrix_c = multiplier(matrix_a, matrix_b)
        return matrix_c

    return wrapper

def vector_multiplication(vector_a: list[int], vector_b: list[int]) -> int:
    """Returns the dot product of two vectors"""
    assert len(vector_a) == len(vector_b)

    product = 0
    for i, ai in enumerate(vector_a):
        product += ai * vector_b[i]

    return product

def print_matrix(matrix: list[list[int]]) -> None:
    """Prints a matrix in a semi-nice format"""
    print("[")
    _ = [print(f"  {i}") for i in matrix]
    print("]")

################################################################################

@matrix_multiplication
def classic_multiplication(
            matrix_a: list[list[int]],
            matrix_b: list[list[int]]
        ) -> list[list[int]]:
    """
    Multiplies two matrices together. The resultant matrix, C, follows that
    C[i][j] = A[i][...] * B[...][j]
    """

    matrix_b_width = get_width(matrix_b)
    matrix_c = [[0 for _ in range(len(matrix_a))] for _ in range(matrix_b_width)]
    for i, vector_a in enumerate(matrix_a):
        j = 0
        while j < matrix_b_width:
            vector_b = [matrix_b[n][j] for n in range(len(matrix_b))]
            dot_product = vector_multiplication(vector_a, vector_b)
            matrix_c[i][j] = dot_product
            j += 1

    return matrix_c

ma = [
    [3, 0, 3, 1],
    [5, 3, 1, 0],
    [3, 0, 3, 1],
    [1, 2, 2, 5]
]

mb = [
    [2, 4, 5, 2],
    [1, 3, 2, 1],
    [3, 0, 5, 0],
    [0, 4, 1, 4]
]

print_matrix(ma)
print("x")
print_matrix(mb)
print("=")
print_matrix(classic_multiplication(ma, mb))
