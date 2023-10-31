import unittest
from collection_types import Vector, Matrix
from main import classic_multiplication, \
                divide_and_conquer_multiplication, \
                    strassen_multiplication, \
                        matrix_equality

class MultiplicationTester(unittest.TestCase):

    def check_case(
                self,
                matrix_a: Matrix,
                matrix_b: Matrix,
                expected: Matrix
            ) -> None:

        classic_result = classic_multiplication(
            matrix_a, matrix_b
        )[0]
        divide_and_conquer_result = divide_and_conquer_multiplication(
            matrix_a, matrix_b
        )[0]
        strassen_result = strassen_multiplication(
            matrix_a, matrix_b
        )[0]

        self.assertTrue(
            matrix_equality(
                classic_result,
                divide_and_conquer_result,
                strassen_result,
                expected
            )
        )

    def testcase_1(self) -> None:
        matrix_a: Matrix = [[1]]
        matrix_b: Matrix = [[1]]
        expected: Matrix = [[1]]

        self.check_case(matrix_a, matrix_b, expected)

    def testcase_2(self) -> None:
        matrix_a: Matrix = [[5]]
        matrix_b: Matrix = [[9]]
        expected: Matrix = [[45]]

        self.check_case(matrix_a, matrix_b, expected)

    def testcase_3(self) -> None:
        matrix_a: Matrix = [[-2]]
        matrix_b: Matrix = [[3]]
        expected: Matrix = [[-6]]

        self.check_case(matrix_a, matrix_b, expected)

    def testcase_4(self) -> None:
        matrix_a: Matrix = [
            [1, 2],
            [3, 4]
        ]
        matrix_b: Matrix = [
            [5, 6],
            [7, 8]
        ]
        expected: Matrix = [
            [19, 22],
            [43, 50]
        ]

        self.check_case(matrix_a, matrix_b, expected)

    def testcase_5(self) -> None:
        matrix_a: Matrix = [
            [99, 99],
            [99, 99]
        ]
        matrix_b: Matrix = [
            [99, 99],
            [99, 99]
        ]
        expected: Matrix = [
            [19_602, 19_602],
            [19_602, 19_602]
        ]

        self.check_case(matrix_a, matrix_b, expected)

    def testcase_6(self) -> None:
        matrix_a: Matrix = [
            [3, 0, 3, 1],
            [5, 3, 1, 0],
            [3, 0, 3, 1],
            [1, 2, 2, 5]
        ]
        matrix_b: Matrix = [
            [2, 4, 5, 2],
            [1, 3, 2, 1],
            [3, 0, 5, 0],
            [0, 4, 1, 4]
        ]
        expected: Matrix = [
            [15, 16, 31, 10],
            [16, 29, 36, 13],
            [15, 16, 31, 10],
            [10, 30, 24, 24]
        ]

        self.check_case(matrix_a, matrix_b, expected)

    def testcase_7(self) -> None:
        matrix_a: Matrix = [
            [-165, -45, 454, 525],
            [454, 56, 659, 56],
            [2_649, 26, -88, 26],
            [1_659, -5, 0, 0]
        ]
        matrix_b: Matrix = [
            [0, 0, 1, 20],
            [161, 1_674, 88, 59],
            [-65, 12, 123, 454],
            [98, -96, 12, 45]
        ]
        expected: Matrix = [
            [14_695, -120_282, 58_017, 223_786],
            [-28_331, 96_276, 87_111, 314_090],
            [12_454, 39_972, -5_575, 15_732],
            [-805, -8_370, 1_219, 32_885]
        ]

        self.check_case(matrix_a, matrix_b, expected)

    def testcase_8(self) -> None:
        matrix_a: Matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ]
        matrix_b: Matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ]
        expected: Matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ]

        self.check_case(matrix_a, matrix_b, expected)

    def testcase_9(self) -> None:
        matrix_a: Matrix = [
            [-165, -45, 454, 525, -165, -45, 454, 525],
            [454, 56, 659, 56, 454, 56, 659, 56],
            [2_649, 26, -88, 26, 2_649, 26, -88, 26],
            [1_659, -5, 0, 0, 1_659, -5, 0, 0],
            [-165, -45, 454, 525, -165, -45, 454, 525],
            [454, 56, 659, 56, 454, 56, 659, 56],
            [2_649, 26, -88, 26, 2_649, 26, -88, 26],
            [1_659, -5, 0, 0, 1_659, -5, 0, 0],
        ]
        matrix_b: Matrix = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ]
        expected: Matrix = [
            [-165, -45, 454, 525, -165, -45, 454, 525],
            [454, 56, 659, 56, 454, 56, 659, 56],
            [2_649, 26, -88, 26, 2_649, 26, -88, 26],
            [1_659, -5, 0, 0, 1_659, -5, 0, 0],
            [-165, -45, 454, 525, -165, -45, 454, 525],
            [454, 56, 659, 56, 454, 56, 659, 56],
            [2_649, 26, -88, 26, 2_649, 26, -88, 26],
            [1_659, -5, 0, 0, 1_659, -5, 0, 0],
        ]

        self.check_case(matrix_a, matrix_b, expected)

    def testcase_10(self) -> None:

        matrix_a: Matrix = [
            [5, 6],
            [7, 8]
        ]
        matrix_b: Matrix = [
            [1, 2],
            [3, 4]
        ]
        expected: Matrix = [
            [23, 34],
            [31, 46]
        ]


        self.check_case(matrix_a, matrix_b, expected)

if __name__ == "__main__":
    unittest.main()