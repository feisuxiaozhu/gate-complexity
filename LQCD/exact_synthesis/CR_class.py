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
        # numerator is an EisensteinInteger and f is a nonnegative integer.
        if not isinstance(numerator, EisensteinInteger):
            raise TypeError("The numerator must be an EisensteinInteger.")
        if f < 0:
            raise ValueError("The exponent f must be a nonnegative integer.")
        self.numerator = numerator
        self.f = f
        self.auto_reduce()

    def auto_reduce(self):
        # Reduce the fraction by dividing the numerator by sqrt(-3) when possible.
        while self.f > 0 and self.numerator.divisible_by_sqrt_neg3():
            self.numerator = self.numerator.divide_by_sqrt_neg3()
            self.f -= 1

    def __repr__(self):
        # Return a string representation of the Eisenstein fraction.
        if self.f == 0:
            return f"{self.numerator}"
        else:
            return f"({self.numerator})/sqrt(-3)^{self.f}"


# Example usage:
# Define omega as the formal element such that omega = (-1+sqrt(-3))/2.
# Here, we represent the Eisenstein integer 1 + 2*omega.
ei = EisensteinInteger(1, 3)
ei2 = EisensteinInteger(1, 3)

# Create an Eisenstein fraction (1+2*omega)/sqrt(-3)^1.
ef = EisensteinFraction(ei*ei2, 100)
print("The fraction (1+2*omega)/sqrt(-3) is reduced to:", ef)
print(ef.numerator)