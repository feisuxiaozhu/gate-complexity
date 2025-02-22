class EisensteinInteger:
    def __init__(self, a, b=0):
        # Represents a + b*omega, with omega = (-1+sqrt(-3))/2.
        self.a = a  # integer coefficient for 1
        self.b = b  # integer coefficient for omega

    def __repr__(self):
        # Return a string representation of the Eisenstein integer.
        if self.b >= 0:
            return f"{self.a} + {self.b}*omega"
        else:
            return f"{self.a} - {-self.b}*omega"

    def __add__(self, other):
        return EisensteinInteger(self.a + other.a, self.b + other.b)

    def __sub__(self, other):
        return EisensteinInteger(self.a - other.a, self.b - other.b)

    def __mul__(self, other):
        # Multiply two Eisenstein integers.
        # (a + b*omega) * (c + d*omega) = (a*c - b*d) + (a*d + b*c - b*d)*omega.
        new_a = self.a * other.a - self.b * other.b
        new_b = self.a * other.b + self.b * other.a - self.b * other.b
        return EisensteinInteger(new_a, new_b)
    
    def __pow__(self, exponent):
        if exponent < 0:
            raise ValueError("Negative exponent is not supported.")
        result = EisensteinInteger(1, 0)
        for _ in range(exponent):
            result = result * self
        return result

    def divisible_by_sqrt_neg3(self):
        # Check if self is divisible by sqrt(-3) = 1 + 2*omega.
        # Write self as a + b*omega. Then self is divisible if
        # (b - 2*a) and (2*b - a) are divisible by 3.
        return ((self.b - 2 * self.a) % 3 == 0) and ((2 * self.b - self.a) % 3 == 0)

    def divide_by_sqrt_neg3(self):
        # Divide self by sqrt(-3) if possible.
        if not self.divisible_by_sqrt_neg3():
            raise ValueError("The Eisenstein integer is not divisible by sqrt(-3).")
        # The quotient is given by:
        # c = (2*b - a) / 3 and d = (b - 2*a) / 3.
        c = (2 * self.b - self.a) // 3
        d = (self.b - 2 * self.a) // 3
        return EisensteinInteger(c, d)

class EisensteinFraction:
    def __init__(self, numerator, f):
        # Numerator must be an EisensteinInteger; f is a nonnegative integer.
        if not isinstance(numerator, EisensteinInteger):
            raise TypeError("The numerator must be an EisensteinInteger.")
        if f < 0:
            raise ValueError("The exponent must be a nonnegative integer.")
        self.numerator = numerator
        self.f = f
        self.auto_reduce()

    def auto_reduce(self):
        # Reduce the fraction by dividing the numerator by sqrt(-3) when possible.
        while self.f > 0 and self.numerator.divisible_by_sqrt_neg3():
            self.numerator = self.numerator.divide_by_sqrt_neg3()
            self.f -= 1

    def __repr__(self):
        if self.f == 0:
            return f"{self.numerator}"
        else:
            return f"({self.numerator})/sqrt(-3)^{self.f}"

    def __add__(self, other):
        if not isinstance(other, EisensteinFraction):
            raise TypeError("Addition is defined for EisensteinFraction objects only.")
        # Use the maximum exponent as a common denominator.
        common_f = max(self.f, other.f)
        # sqrt(-3) is represented by 1 + 2*omega.
        s = EisensteinInteger(1, 2)
        delta_self = common_f - self.f
        delta_other = common_f - other.f
        # Adjust the numerators.
        num_self = self.numerator * (s ** delta_self)
        num_other = other.numerator * (s ** delta_other)
        new_num = num_self + num_other
        return EisensteinFraction(new_num, common_f)

    def __sub__(self, other):
        if not isinstance(other, EisensteinFraction):
            raise TypeError("Subtraction is defined for EisensteinFraction objects only.")
        common_f = max(self.f, other.f)
        s = EisensteinInteger(1, 2)
        delta_self = common_f - self.f
        delta_other = common_f - other.f
        num_self = self.numerator * (s ** delta_self)
        num_other = other.numerator * (s ** delta_other)
        new_num = num_self - num_other
        return EisensteinFraction(new_num, common_f)

    def __mul__(self, other):
        if not isinstance(other, EisensteinFraction):
            raise TypeError("Multiplication is defined for EisensteinFraction objects only.")
        new_num = self.numerator * other.numerator
        new_f = self.f + other.f
        return EisensteinFraction(new_num, new_f)

    def __eq__(self, other):
        if not isinstance(other, EisensteinFraction):
            return False
        common_f = max(self.f, other.f)
        s = EisensteinInteger(1, 2)
        num_self = self.numerator * (s ** (common_f - self.f))
        num_other = other.numerator * (s ** (common_f - other.f))
        return (num_self.a == num_other.a) and (num_self.b == num_other.b)

class EisensteinVector3:
    def __init__(self, x, y, z):
        # Each entry must be an EisensteinFraction.
        if not (isinstance(x, EisensteinFraction) and isinstance(y, EisensteinFraction) and isinstance(z, EisensteinFraction)):
            raise TypeError("Each element must be an EisensteinFraction.")
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"[{self.x}, {self.y}, {self.z}]"

    def __add__(self, other):
        if not isinstance(other, EisensteinVector3):
            raise TypeError("Addition is defined only for EisensteinVector3 instances.")
        return EisensteinVector3(self.x + other.x,
                                 self.y + other.y,
                                 self.z + other.z)

    def __sub__(self, other):
        if not isinstance(other, EisensteinVector3):
            raise TypeError("Subtraction is defined only for EisensteinVector3 instances.")
        return EisensteinVector3(self.x - other.x,
                                 self.y - other.y,
                                 self.z - other.z)

    def dot(self, other):
        # Returns the dot product of two vectors.
        if not isinstance(other, EisensteinVector3):
            raise TypeError("Dot product is defined only for EisensteinVector3 instances.")
        return self.x * other.x + self.y * other.y + self.z * other.z

    def scale(self, scalar):
        # Multiply the vector by an EisensteinFraction scalar.
        if not isinstance(scalar, EisensteinFraction):
            raise TypeError("Scalar must be an EisensteinFraction.")
        return EisensteinVector3(scalar * self.x,
                                 scalar * self.y,
                                 scalar * self.z)

class EisensteinMatrix3x3:
    def __init__(self, matrix):
        # The input matrix should be a list of 3 lists, each containing 3 EisensteinFraction objects.
        if len(matrix) != 3:
            raise ValueError("Matrix must have 3 rows.")
        for row in matrix:
            if len(row) != 3:
                raise ValueError("Each row must have 3 columns.")
            for elem in row:
                if not isinstance(elem, EisensteinFraction):
                    raise TypeError("Each element must be an EisensteinFraction.")
        self.matrix = matrix

    def __repr__(self):
        rows = []
        for row in self.matrix:
            rows.append("[" + ", ".join([str(elem) for elem in row]) + "]")
        return "\n".join(rows)

    def __add__(self, other):
        if not isinstance(other, EisensteinMatrix3x3):
            raise TypeError("Addition is defined only for EisensteinMatrix3x3 objects.")
        new_matrix = []
        for i in range(3):
            new_row = []
            for j in range(3):
                new_row.append(self.matrix[i][j] + other.matrix[i][j])
            new_matrix.append(new_row)
        return EisensteinMatrix3x3(new_matrix)

    def __sub__(self, other):
        if not isinstance(other, EisensteinMatrix3x3):
            raise TypeError("Subtraction is defined only for EisensteinMatrix3x3 objects.")
        new_matrix = []
        for i in range(3):
            new_row = []
            for j in range(3):
                new_row.append(self.matrix[i][j] - other.matrix[i][j])
            new_matrix.append(new_row)
        return EisensteinMatrix3x3(new_matrix)

    def __mul__(self, other):
        # If other is a scalar (EisensteinFraction), scale the matrix.
        if isinstance(other, EisensteinFraction):
            new_matrix = []
            for row in self.matrix:
                new_row = [elem * other for elem in row]
                new_matrix.append(new_row)
            return EisensteinMatrix3x3(new_matrix)
        # If other is another 3-by-3 matrix, perform matrix multiplication.
        elif isinstance(other, EisensteinMatrix3x3):
            new_matrix = []
            for i in range(3):
                new_row = []
                for j in range(3):
                    # Compute the (i,j) entry as the dot product of row i of self and column j of other.
                    sum_elem = None
                    for k in range(3):
                        product = self.matrix[i][k] * other.matrix[k][j]
                        if sum_elem is None:
                            sum_elem = product
                        else:
                            sum_elem = sum_elem + product
                    new_row.append(sum_elem)
                new_matrix.append(new_row)
            return EisensteinMatrix3x3(new_matrix)
        # If other is a 3-by-1 vector of EisensteinFraction.
        elif isinstance(other, EisensteinVector3):
            new_x = (self.matrix[0][0] * other.x +
                     self.matrix[0][1] * other.y +
                     self.matrix[0][2] * other.z)
            new_y = (self.matrix[1][0] * other.x +
                     self.matrix[1][1] * other.y +
                     self.matrix[1][2] * other.z)
            new_z = (self.matrix[2][0] * other.x +
                     self.matrix[2][1] * other.y +
                     self.matrix[2][2] * other.z)
            return EisensteinVector3(new_x, new_y, new_z)
        else:
            raise TypeError("Multiplication is defined only for EisensteinFraction, EisensteinMatrix3x3, or EisensteinVector3 objects.")

def HMatrix():
    omega = EisensteinFraction(EisensteinInteger(0, 1), 0)
    omega2 = omega * omega
    one_over_sqrt_neg3 = EisensteinFraction(EisensteinInteger(1, 0), 1)

    H_numerator = EisensteinMatrix3x3([
        [
            EisensteinFraction(EisensteinInteger(1, 0), 0),
            EisensteinFraction(EisensteinInteger(1, 0), 0),
            EisensteinFraction(EisensteinInteger(1, 0), 0)
        ],
        [
            EisensteinFraction(EisensteinInteger(1, 0), 0),
            omega,
            omega2
        ],
        [
            EisensteinFraction(EisensteinInteger(1, 0), 0),
            omega2,
            omega
        ]
    ])
    H = H_numerator * one_over_sqrt_neg3
    return H

def SMatrix():
    S = EisensteinMatrix3x3([
        [
            EisensteinFraction(EisensteinInteger(1, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0)
        ],
        [
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            omega,
            EisensteinFraction(EisensteinInteger(0, 0), 0)
        ],
        [
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(1, 0), 0)
        ]
    ])
    return S

def RMatrix():
    R = EisensteinMatrix3x3([
        [
            EisensteinFraction(EisensteinInteger(1, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0)
        ],
        [
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(1, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0)
        ],
        [
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(-1, 0), 0)
        ]
    ])
    return R

def XMatrix():
    X = EisensteinMatrix3x3([
        [
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(1, 0), 0)
        ],
        [
            EisensteinFraction(EisensteinInteger(1, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0)
        ],
        [
            EisensteinFraction(EisensteinInteger(0, 0), 0),
            EisensteinFraction(EisensteinInteger(1, 0), 0),
            EisensteinFraction(EisensteinInteger(0, 0), 0)
        ]
    ])
    return X

def DMatrix(a, b, c):
    # Suppose omega is an EisensteinInteger(0, 1).
    # Define a zero fraction for the off-diagonal entries.
    zero_fraction = EisensteinFraction(EisensteinInteger(0, 0), 0)

    # Compute omega^a, omega^b, omega^c as Eisenstein integers,
    # then make each one into an Eisenstein fraction with exponent 0.
    omega_a = EisensteinFraction(EisensteinInteger(0, 1) ** a, 0)
    omega_b = EisensteinFraction(EisensteinInteger(0, 1) ** b, 0)
    omega_c = EisensteinFraction(EisensteinInteger(0, 1) ** c, 0)

    # Build the diagonal matrix as a 3-by-3 matrix of EisensteinFraction entries.
    return EisensteinMatrix3x3([
        [omega_a, zero_fraction, zero_fraction],
        [zero_fraction, omega_b, zero_fraction],
        [zero_fraction, zero_fraction, omega_c]
    ])


# dmat = DMatrix(1, 2, 3)
# print("DMatrix(1, 2, 3) =")
# print(dmat)



# print("H =")
# print(H)
# print("\nS =")
# print(S)
# print("\nR =")
# print(R)
# print("\nX =")
# print(X)
# Example usage:
# It is assumed that EisensteinFraction, EisensteinInteger, and EisensteinVector3 have been defined.

# Create a zero fraction and a one fraction.
# zero_ei = EisensteinInteger(0, 0)
# zero_fraction = EisensteinFraction(zero_ei, 0)

# one_ei = EisensteinInteger(1, 2)
# one_fraction = EisensteinFraction(one_ei, 1)

# # Define a 3-by-3 matrix similar to the identity matrix.
# matrix = EisensteinMatrix3x3([
#     [one_fraction, zero_fraction, zero_fraction],
#     [zero_fraction, one_fraction, zero_fraction],
#     [zero_fraction, zero_fraction, one_fraction]
# ])



# # Define a 3-by-1 vector.
# vector = EisensteinVector3(one_fraction, zero_fraction, zero_fraction)
# print(matrix, vector)

# # Multiply the matrix by the vector.
# result_vector = matrix * vector

# print("The product of the matrix and the vector is:")
# print(result_vector)




# Example usage:
# Define omega as the formal element such that omega = (-1+sqrt(-3))/2.
# Here, we represent the Eisenstein integer 1 + 2*omega.
# ei = EisensteinInteger(1, 3)
# ei2 = EisensteinInteger(1, 3)

# # Create an Eisenstein fraction (1+2*omega)/sqrt(-3)^1.
# ef = EisensteinFraction(ei*ei2, 100)
# print("The fraction (1+2*omega)/sqrt(-3) is reduced to:", ef)
# print(ef.numerator)

# ei1 = EisensteinInteger(1, 2)
# ef1 = EisensteinFraction(ei1, 1)  # This will auto-reduce to 1 since (1+2*omega)/sqrt(-3)=1.

# ei2 = EisensteinInteger(3, -1)
# ef2 = EisensteinFraction(ei2, 0)

# ei3 = EisensteinInteger(0, 1)
# ef3 = EisensteinFraction(ei3, 0)

# # Create an Eisenstein vector with these fractions.
# v = EisensteinVector3(ef1, ef2, ef3)
# print("The Eisenstein vector is:", v)




# zero_ei = EisensteinInteger(0, 0)
# zero_fraction = EisensteinFraction(zero_ei, 0)

# one_ei = EisensteinInteger(1, 0)
# one_fraction = EisensteinFraction(one_ei, 0)

# # Create a 3-by-3 matrix similar to the identity matrix.
# matrix = EisensteinMatrix3x3([
#     [one_fraction, zero_fraction, zero_fraction],
#     [zero_fraction, one_fraction, zero_fraction],
#     [zero_fraction, zero_fraction, one_fraction]
# ])

# print("The 3-by-3 matrix is:")
# print(matrix)