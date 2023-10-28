"""
Implements methods for classic, d&q, and strassen matrix multiplication
"""
# pylint: disable=unbalanced-tuple-unpacking
from typing import Callable, Any
from random import randint
from time import time
from matplotlib import pyplot as plt

type Vector = list[int]
type Matrix = list[Vector]

# VECTOR OPERATIONS ############################################################
def vector_multiplication(vector_a: Vector, vector_b: Vector) -> int:
    """Returns the dot product of two vectors"""
    assert len(vector_a) == len(vector_b)

    product = 0
    for i, ai in enumerate(vector_a):
        product += ai * vector_b[i]

    return product

def vector_addition(vector_a: Vector, vector_b: Vector) -> Vector:
    """Adds two vectors together"""
    assert len(vector_a) == len(vector_b)

    vector_c = []
    for i, ai in enumerate(vector_a):
        vector_c.append(ai + vector_b[i])

    return vector_c

def vector_negation(vector: Vector) -> Vector:
    """Returns the same vector but with all the values negated"""
    neg_vector = []
    for i in vector:
        neg_vector.append(i * -1)
    return neg_vector

def vector_equality(*vectors: Vector):
    """Returns True if all vectors are equal"""
    if not vectors:
        return True

    first = vectors[0]
    for vector in vectors[1:]:
        if not first == vector:
            return False

    return True

# MATRIX OPERATIONS ############################################################
def matrix_multiplication(
            multiplier: Callable[
                            [Matrix, Matrix],
                            Matrix
                        ]
        ):
    """A decorator for a function taking two matrices and returning a matrix"""

    def wrapper(
                matrix_a: Matrix,
                matrix_b: Matrix,
                root_call = True
            ):
        """
        Procedure to take place any time a matrix multiplication function
        is called
        """
        start_time = time()

        assert can_multiply(matrix_a, matrix_b)
        matrix_c = multiplier(matrix_a, matrix_b)

        duration = time() - start_time

        if root_call:
            print(f"({multiplier.__name__}) Time Taken: {duration:.8f} seconds")

        return matrix_c, duration

    return wrapper

def matrix_addition(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    """Adds two matrices together"""
    assert len(matrix_a) == len(matrix_b)

    matrix_c = []
    for i, vector_a in enumerate(matrix_a):
        vector_b = matrix_b[i]
        matrix_c.append(vector_addition(vector_a, vector_b))

    return matrix_c

def matrix_negation(matrix: Matrix) -> Matrix:
    """Returns the same matrix but with all the vectors negated"""
    neg_matrix = []
    for vector in matrix:
        neg_matrix.append(vector_negation(vector))
    return neg_matrix

def matrix_equality(*matrices: Matrix):
    """Returns True if all matrices are equal"""
    if not matrices:
        return True

    first = matrices[0]
    for matrix in matrices[1:]:
        if not first == matrix:
            return False

    return True

# MATRIX HELPERS ###############################################################
def get_width(matrix: Matrix) -> int:
    """Returns the width of the matrix"""
    if not matrix:
        return 0
    return len(matrix[0])

def can_multiply(
            matrix_a: Matrix,
            matrix_b: Matrix
        ) -> bool:
    """Denotes whether or not two matrices can be multiplied"""
    return get_width(matrix_a) == len(matrix_b)

def split_matrix(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    """Splits a Matrix into 4 even quadrant matrices"""
    m11, m12, m21, m22 = [], [], [], []

    matrix_len = len(matrix)
    matrix_wid = get_width(matrix)
    row_mid = matrix_len // 2
    col_mid = matrix_wid // 2

    for i, row in enumerate(matrix):
        if i < col_mid:
            m11.append(row[0:row_mid])
            m12.append(row[row_mid:matrix_wid])
        else:
            m21.append(row[0:row_mid])
            m22.append(row[row_mid:matrix_wid])

    return m11, m12, m21, m22

def combine_matrix(
            m11: Matrix,
            m12: Matrix,
            m21: Matrix,
            m22: Matrix
        ) -> Matrix:
    """Combines 4 quadrants of a Matrix into one"""
    matrix = []
    offset = len(m11)
    for i in range(offset + len(m21)):
        if i < offset:
            matrix.append(m11[i] + m12[i])
        else:
            matrix.append(m21[i - offset] + m22[i - offset])

    return matrix

def generate_random_matrix(min_n: int, max_n: int, dim: int):
    """
    Generates a random matrix of dimensions dim x dim with ints ranging from
    min_n to max_n (using the random.randint function)
    """
    return [[randint(min_n, max_n) for _ in range(dim)] for _ in range(dim)]

def initialize_matrix(n_rows: int, n_cols: int) -> Matrix:
    """Initializes an empty int matrix given its dimensions"""
    return [[0 for _ in range(n_rows)] for _ in range(n_cols)]

def print_matrix(*args: Any, matrix: Matrix) -> None:
    """Prints a matrix in a semi-nice format"""
    print(*args, "\n[")
    _ = [print(" ", " ".join([str(n) for n in i])) for i in matrix]
    print("]")

# MATRIX MULTIPLIERS ###########################################################
@matrix_multiplication
def classic_multiplication(
            matrix_a: Matrix,
            matrix_b: Matrix
        ) -> Matrix:
    """
    Multiplies two matrices together. The resultant matrix, C, follows that
    C[i][j] = A[i][...] * B[...][j]
    """
    matrix_b_width = get_width(matrix_b)
    matrix_c = initialize_matrix(len(matrix_a), matrix_b_width)
    for i, vector_a in enumerate(matrix_a):
        for j in range(matrix_b_width):
            vector_b = [matrix_b[n][j] for n in range(len(matrix_b))]
            dot_product = vector_multiplication(vector_a, vector_b)
            matrix_c[i][j] = dot_product
    return matrix_c

@matrix_multiplication
def divide_and_conquer_multiplication(
            matrix_a: Matrix,
            matrix_b: Matrix
        ) -> Matrix:
    """
    Multiplies two matrices together. The resultant matrix, C, follows that
    C[i][j] = A[i][...] * B[...][j]
    """
    if len(matrix_a) == get_width(matrix_a) \
            == len(matrix_b) == get_width(matrix_b) \
                <= 2:
        return classic_multiplication(matrix_a, matrix_b, root_call=False)[0]

    a11, a12, a21, a22 = split_matrix(matrix_a)
    b11, b12, b21, b22 = split_matrix(matrix_b)

    c11 = matrix_addition(
        divide_and_conquer_multiplication(a11, b11, root_call=False)[0],
        divide_and_conquer_multiplication(a12, b21, root_call=False)[0]
    )
    c12 = matrix_addition(
        divide_and_conquer_multiplication(a11, b12, root_call=False)[0],
        divide_and_conquer_multiplication(a12, b22, root_call=False)[0]
    )
    c21 = matrix_addition(
        divide_and_conquer_multiplication(a21, b11, root_call=False)[0],
        divide_and_conquer_multiplication(a22, b21, root_call=False)[0]
    )
    c22 = matrix_addition(
        divide_and_conquer_multiplication(a21, b12, root_call=False)[0],
        divide_and_conquer_multiplication(a22, b22, root_call=False)[0]
    )

    return combine_matrix(c11, c12, c21, c22)

@matrix_multiplication
def strassen_multiplication(
            matrix_a: Matrix,
            matrix_b: Matrix
        ) -> Matrix:
    """
    Multiplies two matrices together. The resultant matrix, C, follows that
    C[i][j] = A[i][...] * B[...][j]
    """
    if len(matrix_a) == get_width(matrix_a) \
            == len(matrix_b) == get_width(matrix_b) \
                == 1:
        return [[matrix_a[0][0] * matrix_b[0][0]]]

    if len(matrix_a) == get_width(matrix_a) \
            == len(matrix_b) == get_width(matrix_b) \
                == 2:
        return classic_multiplication(matrix_a, matrix_b, root_call=False)[0]

    a11, a12, a21, a22 = split_matrix(matrix_a)
    b11, b12, b21, b22 = split_matrix(matrix_b)

    p = strassen_multiplication(
        matrix_addition(a11, a22),
        matrix_addition(b11, b22),
        root_call=False
    )[0]
    q = strassen_multiplication(
        matrix_addition(a21, a22),
        b11,
        root_call=False
    )[0]
    r = strassen_multiplication(
        a11,
        matrix_addition(b12, matrix_negation(b22)),
        root_call=False
    )[0]
    s = strassen_multiplication(
        a22,
        matrix_addition(b21, matrix_negation(b11)),
        root_call=False
    )[0]
    t = strassen_multiplication(
        matrix_addition(a11, a12),
        b22,
        root_call=False
    )[0]
    u = strassen_multiplication(
        matrix_addition(a21, matrix_negation(a11)),
        matrix_addition(b11, b12),
        root_call=False
    )[0]
    v = strassen_multiplication(
        matrix_addition(a12, matrix_negation(a22)),
        matrix_addition(b21, b22),
        root_call=False
    )[0]

    c11 = matrix_addition( # P + S - T + V
        matrix_addition(matrix_addition(p, s), matrix_negation(t)), v
    )
    c12 = matrix_addition(r, t) # R + T
    c21 = matrix_addition(q, s) # Q + S
    c22 = matrix_addition( # P + R - Q + U
        matrix_addition(matrix_addition(p, r), matrix_negation(q)), u
    )

    return combine_matrix(c11, c12, c21, c22)

# DRIVER #######################################################################
FILE_NAME = "comparison"
DIMENSIONS = 1

to_plot = []
while True:
    try:
        start = time()

        # Create random input matrices
        print(f"Matrices with dim {DIMENSIONS} are being created!")
        ma: Matrix = generate_random_matrix(1, 50, DIMENSIONS)
        # print_matrix("MA = ", matrix=ma)
        mb: Matrix = generate_random_matrix(1, 50, DIMENSIONS)
        # print_matrix("MB = ", matrix=mb)
        print("Beginning Multiplication.")

        mcc, c_duration = classic_multiplication(ma, mb)
        # print_matrix("MCC = ", matrix=mcc)

        mcd, d_duration = divide_and_conquer_multiplication(ma, mb)
        # print_matrix("MCD = ", matrix=mcd)

        mcs, s_duration = strassen_multiplication(ma, mb)
        # print_matrix("MCS = ", matrix=mcs)

        # Store the durations
        to_plot.append((DIMENSIONS, c_duration, d_duration, s_duration))
        print(to_plot, file=open(f"{FILE_NAME}.txt", 'w', encoding="utf-8"))

        # Make sure we got the same answer for all three algorithms
        assert matrix_equality(mcc, mcd, mcs)
        print("All three calculations returned the same matrix!\n\n")

    # If something goes wrong, exit so we can at least plot it
    # This also lets us Ctrl+C to exit and still keep/plot the data
    except:
        break

    DIMENSIONS *= 2

    # If we're taking longer than 8 hours to get results
    # if time() - start > 8 * 3_600:
    #    break

print(to_plot, file=open(f"{FILE_NAME}.txt", 'w', encoding="utf-8"))

dimensions_used = [i[0] for i in to_plot]
c_durations = [i[1]//3600 for i in to_plot]
d_durations = [i[2]//3600 for i in to_plot]
s_durations = [i[3]//3600 for i in to_plot]

plt.title("Multiplication Algorithm Comparison")
plt.xlabel("Dimensions Used")
plt.ylabel("Time Taken (h)")

plt.plot(dimensions_used, c_durations, color="red", label="Classic")
plt.plot(dimensions_used, d_durations, color="green", label="Divide & Conquer")
plt.plot(dimensions_used, s_durations, color="blue", label="Strassen")

plt.legend()

plt.savefig(f"{FILE_NAME}.png")
plt.show()
