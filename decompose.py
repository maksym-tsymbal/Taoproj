import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

s = sp.symbols('s')


# ===============================
# Парсинг передатної функції
# ===============================
def parse_tf(expr_str):

    transformations = standard_transformations + (
        implicit_multiplication_application,
    )

    expr = parse_expr(expr_str.replace("^", "**"),
                      transformations=transformations)

    num, den = sp.fraction(expr)

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_c = np.array([float(c) for c in num_poly.all_coeffs()])
    den_c = np.array([float(c) for c in den_poly.all_coeffs()])

    return expr, num_c, den_c


# ===============================
# Розклад на ланки
# ===============================
def decompose_into_blocks(num_c, den_c):

    zeros = np.roots(num_c)
    poles = np.roots(den_c)

    blocks = []

    # коефіцієнт підсилення
    K = abs(num_c[-1] / den_c[-1])
    blocks.append(("K", K))

    # ---- НУЛІ ----
    used = [False]*len(zeros)

    for i, z in enumerate(zeros):

        if used[i]:
            continue

        if abs(z.imag) < 1e-6:
            if abs(z.real) < 1e-8:
                blocks.append(("s", None))
            else:
                blocks.append(("zero", abs(z.real)))
            used[i] = True

        else:
            for j in range(i+1, len(zeros)):
                if not used[j] and abs(zeros[j] - np.conj(z)) < 1e-6:
                    blocks.append(("osc_zero", abs(z)))
                    used[i] = True
                    used[j] = True
                    break

    # ---- ПОЛЮСИ ----
    used = [False]*len(poles)

    for i, p in enumerate(poles):

        if used[i]:
            continue

        if abs(p.imag) < 1e-6:
            if abs(p.real) < 1e-8:
                blocks.append(("integrator", None))
            else:
                blocks.append(("pole", abs(p.real)))
            used[i] = True

        else:
            for j in range(i+1, len(poles)):
                if not used[j] and abs(poles[j] - np.conj(p)) < 1e-6:
                    blocks.append(("osc_pole", abs(p)))
                    used[i] = True
                    used[j] = True
                    break

    return blocks


# ===============================
# Текстовий вигляд ланки
# ===============================
def block_to_string(block):

    kind, val = block

    if kind == "K":
        return f"{val:.4g}"

    elif kind == "zero":
        return f"(1 + s/{val:.4g})"

    elif kind == "pole":
        return f"1/(1 + s/{val:.4g})"

    elif kind == "integrator":
        return "1/s"

    elif kind == "s":
        return "s"

    elif kind == "osc_pole":
        return f"1/(1 + 2ζ(s/{val:.4g}) + (s/{val:.4g})²)"

    elif kind == "osc_zero":
        return f"(1 + 2ζ(s/{val:.4g}) + (s/{val:.4g})²)"

    return "?"


# ===============================
# ТЕСТ
# ===============================
if __name__ == "__main__":

    expr_str = input("Введи Wp(s): ")

    expr, num_c, den_c = parse_tf(expr_str)

    blocks = decompose_into_blocks(num_c, den_c)

    print("\n=== Розклад на ланки ===\n")

    for i, block in enumerate(blocks, 1):
        print(f"W{i}(s): {block_to_string(block)}")