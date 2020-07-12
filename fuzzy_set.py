from utils import clip


class FuzzySet:
    def __init__(self, x_zero, x_one, sink_beyond_1=True, y_symmetry=False):
        x_zero = abs(x_zero) if y_symmetry else x_zero
        x_one = abs(x_one) if y_symmetry else x_one
        self.a = 1 / (x_one - x_zero)
        self.b = -self.a * x_zero
        self.sink_beyond_1 = sink_beyond_1
        self.y_symmetry = y_symmetry
        self.x_min = min(x_zero, x_one) if not y_symmetry else -min(abs(x_zero), abs(x_one))
        self.x_max = max(x_zero, x_one) if not y_symmetry else max(abs(x_zero), abs(x_one))

    def fuzzify(self, val):
        val = abs(val) if self.y_symmetry else val
        y = self.a * val + self.b
        if y > 1 and self.sink_beyond_1:
            return 0
        else:
            return clip(y, 0, 1)

    def defuzzify(self, val):
        assert 0 <= val <= 1
        return (val - self.b) / self.a


if __name__ == "__main__":  # Only for testing purposes
    f_var = FuzzySet(-0.3, -3.0, False, False)
    print(f_var.fuzzify(0.4))
    print(f_var.fuzzify(-4.9))
    print(f_var.defuzzify(0.9))

