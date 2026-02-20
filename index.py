import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import signal
import random
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

s = sp.symbols('s')

# ===============================
# Парсинг Wp(s)
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

    num_c = [float(c) for c in num_poly.all_coeffs()]
    den_c = [float(c) for c in den_poly.all_coeffs()]
    return expr, num_c, den_c


# ===============================
# Розклад на ланки
# ===============================
def decompose_into_blocks(num_c, den_c):
    zeros = np.roots(num_c)
    poles = np.roots(den_c)

    blocks = []

    # коефіцієнт підсилення
    K = num_c[0] / den_c[0]
    blocks.append(("K", K))

    for z in zeros:
        if abs(z.imag) < 1e-6:
            if abs(z.real) < 1e-8:
                blocks.append(("s", None))
            else:
                blocks.append(("zero", abs(z.real)))

    for p in poles:
        if abs(p.imag) < 1e-6:
            if abs(p.real) < 1e-8:
                blocks.append(("integrator", None))
            else:
                blocks.append(("pole", abs(p.real)))
        else:
            blocks.append(("osc_pole", abs(p)))

    return blocks


# ===============================
# Асимптота ОДНІЄЇ ланки
# ===============================
def asymptotic_block(block, w):
    kind, val = block
    mag = np.zeros_like(w)

    if kind == "K":
        mag[:] = 20 * np.log10(abs(val))

    elif kind == "zero":      # (Ts+1)
        w0 = val
        mag[w >= w0] = 20 * np.log10(w[w >= w0] / w0)

    elif kind == "pole":      # 1/(Ts+1)
        w0 = val
        mag[w >= w0] = -20 * np.log10(w[w >= w0] / w0)

    elif kind == "integrator":  # 1/s
        mag = -20 * np.log10(w)

    elif kind == "s":          # s
        mag = 20 * np.log10(w)

    elif kind == "osc_pole":   # коливальна ланка
        w0 = val
        mag[w >= w0] = -40 * np.log10(w[w >= w0] / w0)

    return mag


# ===============================
# Перевірка на максимум
# ===============================
def has_maximum(mag):
    for i in range(1, len(mag) - 1):
        if mag[i] > mag[i - 1] and mag[i] > mag[i + 1]:
            return True
    return False


# ===============================
# Бажана ЛАЧХ
# ===============================
def build_desired_lachh(w, w3):
    L = np.zeros_like(w)

    w1 = w3 / 10     # 1 декада вниз
    w2 = w3 * 10     # 1 декада вгору

    for i, wi in enumerate(w):
        if wi < w1:
            L[i] = -20 * np.log10(wi / w1)
        elif wi < w2:
            L[i] = -20 * np.log10(wi / w3)
        else:
            L[i] = (-20 * np.log10(w2 / w3)
                    -40 * np.log10(wi / w2))
    return L


# ===============================
# main
# ===============================
def main():
    print("=== СИНТЕЗ МЕТОДОМ ЛАЧХ (ТЕСТОВИЙ СКРИПТ) ===")
    print("Приклад:")
    print("(0.25s+0.04)/(150s^3+6.5s^2+1.05s+0.01)\n")

    expr_str = input("Введи Wp(s): ")
    expr, num_c, den_c = parse_tf(expr_str)

    tf = signal.TransferFunction(num_c, den_c)
    blocks = decompose_into_blocks(num_c, den_c)

    print("\n=== ЛАНКИ ===")
    for i, (k, v) in enumerate(blocks, 1):
        if v is None:
            print(f"W{i}(s): {k}")
        else:
            print(f"W{i}(s): {k}, ω0 = {v:.4g}")

    w = np.logspace(-3, 3, 3000)

    # точна ЛАЧХ
    _, mag_exact, _ = signal.bode(tf, w)

    # асимптоти
    mag_sum = np.zeros_like(w)

    plt.figure(figsize=(9, 6))
    colors = plt.cm.tab10.colors

    for i, block in enumerate(blocks):
        mag_i = asymptotic_block(block, w)
        mag_sum += mag_i
        plt.semilogx(
            w, mag_i, ":",
            color=colors[i % len(colors)],
            label=f"Ланка {i+1}"
        )

    plt.semilogx(w, mag_sum, "k--", lw=2,
                 label="Результуюча асимптотична ЛАЧХ")
    plt.semilogx(w, mag_exact, color="steelblue",
                 lw=1.5, label="Точна ЛАЧХ")

    plt.grid(True, which="both")
    plt.xlabel("ω, рад/с")
    plt.ylabel("L(ω), дБ")
    plt.title("Асимптотичні ЛАЧХ кожної ланки та розімкненої САУ")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===== Аналіз типу ЛАЧХ =====
    print("\n=== АНАЛІЗ ЛАЧХ ===")
    if has_maximum(mag_sum):
        system_type = "max"
        print("ЛАЧХ має максимум → ПРАВА номограма")
    else:
        system_type = "mono"
        print("ЛАЧХ монотонна → ЛІВА номограма")

    # ===== Ввід tp =====
    tp = float(input("\nВведи час регулювання tp [c]: "))

    # ===== Вибір σm та ω3tp =====
    if system_type == "max":
        sigma = 6
        w3tp = random.choice([6, 6.5, 7])
        print(f"Прийнято σm = {sigma} %")
        print(f"Прийнято ω3·tp = {w3tp}")

    else:
        sigma = random.choice([5, 6, 7])
        print(f"Прийнято σm = {sigma} %")
        w3tp = float(input("Введи ω3·tp з номограми: "))

    w3 = w3tp / tp
    print(f"\nРозрахована частота зрізу ω3 = {w3:.4g} рад/с")

    # ===== Бажана ЛАЧХ =====
    Lb = build_desired_lachh(w, w3)

    plt.figure(figsize=(8, 5))
    plt.semilogx(w, Lb, "k", lw=2)
    plt.grid(True, which="both")
    plt.xlabel("ω, рад/с")
    plt.ylabel("L_b(ω), дБ")
    plt.title("Бажана логарифмічна АЧХ")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
