import numpy as np
import sympy as sp
from scipy.optimize import minimize

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
# Парсинг з правильною нормалізацією
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
# Нормалізація (головне виправлення)
# ===============================
def normalize_tf(num_c, den_c):
    lead = den_c[0]
    num_n = [c / lead for c in num_c]
    den_n = [c / lead for c in den_c]
    return num_n, den_n

def decompose_into_blocks(num_c, den_c):

    zeros = np.roots(num_c)
    poles = np.roots(den_c)

    blocks = []

    # ---- КОЕФІЦІЄНТ ПІДСИЛЕННЯ ----
    K = abs(num_c[-1] / den_c[-1])

    blocks.append(("K", K))

    # ===============================
    # ОБРОБКА НУЛІВ
    # ===============================
    used = [False]*len(zeros)

    for i, z in enumerate(zeros):

        if used[i]:
            continue

        if abs(z.imag) < 1e-6:
            # реальний нуль
            if abs(z.real) < 1e-8:
                blocks.append(("s", None))
            else:
                blocks.append(("zero", abs(z.real)))

            used[i] = True

        else:
            # комплексна пара
            for j in range(i+1, len(zeros)):
                if not used[j] and abs(zeros[j] - np.conj(z)) < 1e-6:
                    blocks.append(("osc_zero", abs(z)))
                    used[i] = True
                    used[j] = True
                    break


    # ===============================
    # ОБРОБКА ПОЛЮСІВ
    # ===============================
    used = [False]*len(poles)

    for i, p in enumerate(poles):

        if used[i]:
            continue

        if abs(p.imag) < 1e-6:
            # реальний полюс
            if abs(p.real) < 1e-8:
                blocks.append(("integrator", None))
            else:
                blocks.append(("pole", abs(p.real)))

            used[i] = True

        else:
            # комплексна пара
            for j in range(i+1, len(poles)):
                if not used[j] and abs(poles[j] - np.conj(p)) < 1e-6:
                    blocks.append(("osc_pole", abs(p)))
                    used[i] = True
                    used[j] = True
                    break

    return blocks

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


def asymptotic_block(block, w):

    kind, val = block
    mag = np.zeros_like(w)

    if kind == "K":
        mag[:] = 20 * np.log10(abs(val))

    elif kind == "zero":  # (Ts+1)
        w0 = val
        mag[w >= w0] = 20 * np.log10(w[w >= w0] / w0)

    elif kind == "pole":  # 1/(Ts+1)
        w0 = val
        mag[w >= w0] = -20 * np.log10(w[w >= w0] / w0)

    elif kind == "integrator":
        mag = -20 * np.log10(w)

    elif kind == "s":
        mag = 20 * np.log10(w)

    elif kind == "osc_pole":  # 2-го порядку
        w0 = val
        mag[w >= w0] = -40 * np.log10(w[w >= w0] / w0)

    elif kind == "osc_zero":
        w0 = val
        mag[w >= w0] = 40 * np.log10(w[w >= w0] / w0)

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

    w1 = w3 / 10
    w2 = w3 * 10

    for i, wi in enumerate(w):

        # НЧ область
        if wi < w1:
            L[i] = -20 * np.log10(wi / w3)

        # СЧ область
        elif wi <= w2:
            L[i] = -20 * np.log10(wi / w3)

        # ВЧ область
        else:
            L[i] = (
                -20 * np.log10(w2 / w3)
                -40 * np.log10(wi / w2)
            )

    return L

def compute_slope(L, w):
    logw = np.log10(w)
    slope = np.gradient(L, logw)
    return slope


def detect_breakpoints(Lk, w):

    slope = compute_slope(Lk, w)
    slope_diff = np.gradient(slope)

    threshold = 15   # поріг чутливості
    idx = np.where(np.abs(slope_diff) > threshold)[0]

    # залишаємо тільки унікальні точки
    w_breaks = []
    for i in idx:
        if len(w_breaks) == 0 or abs(np.log10(w[i]) - np.log10(w_breaks[-1])) > 0.2:
            w_breaks.append(w[i])

    return w_breaks


# ===============================
# РУЧНИЙ МЕТОДИЧНИЙ СИНТЕЗ
# ===============================
def synthesize_from_breakpoints():

    s = sp.symbols('s')

    print("\n=== СИНТЕЗ Wk(s) ЗА ВВЕДЕНИМИ ПЕРЕЛОМАМИ ===")

    n = int(input("Кількість переломів: "))

    current_slope = 0
    Wk = 1   # <-- ІНІЦІАЛІЗАЦІЯ ТУТ

    for i in range(n):

        print(f"\nПерелом {i+1}")

        w = float(input("ω = "))
        slope_after = float(input("Нахил після перелому (дБ/дек): "))

        delta = slope_after - current_slope
        current_slope = slope_after

        T = 1 / w

        if delta == -20:
            print("→ Інтегратор")
            Wk *= 1/(T*s)

        elif delta == 20:
            print("→ Форсуюча 1-го порядку")
            Wk *= (T*s + 1)

        elif delta == 40:
            print("→ Форсуюча 2-го порядку")
            Wk *= (T**2*s**2 + 2*T*s + 1)

        elif delta == -40:
            print("→ Коливальна 2-го порядку")
            Wk *= 1/(T**2*s**2 + 2*T*s + 1)

        else:
            print("⚠ Нестандартна зміна нахилу")

    print("\nПередатна функція Wk(s):\n")
    sp.pprint(Wk, use_unicode=True)

    return Wk





def metodical_synthesis():

    s = sp.symbols('s')

    print("\n=== МЕТОДИЧНИЙ СИНТЕЗ КОРЕГУЮЧОГО ПРИСТРОЮ ===")

    # Введення переломів
    w1 = float(input("ω1 (інтегратор): "))
    w2 = float(input("ω2 (-20 → 0): "))
    w3 = float(input("ω3 (0 → +40): "))
    w4 = float(input("ω4 (+40 → 0): "))
    xi = float(input("ξ (коеф. коливальності, напр. 0.5 або 0.95): "))

    T1 = 1 / w1
    T2 = 1 / w2
    T3 = 1 / w3
    T4 = 1 / w4

    # === ТОЧНА МОДЕЛЬ ===
    Wk_exact = (1/(T1*s)) \
        * (T2*s + 1) \
        * (T3**2*s**2 + 2*xi*T3*s + 1) \
        * 1/(T4**2*s**2 + 2*xi*T4*s + 1)

    print("\n--- Повна модель Wk(s) ---\n")
    sp.pprint(Wk_exact, use_unicode=True)

    # === СПРОЩЕНА МОДЕЛЬ (як в методичці) ===
    Wk_approx = (1/(T1*s)) \
        * (T3**2*s**2 + 2*xi*T3*s + 1) \
        * 1/(T4*s + 1)

    print("\n--- Спрощена модель Wk_approx(s) ---\n")
    sp.pprint(Wk_approx, use_unicode=True)

    return Wk_exact, Wk_approx


def plot_bode_sympy(W_expr, title):

    s = sp.symbols('s')

    W_expr = sp.together(W_expr)  # звести до одного дробу
    num, den = sp.fraction(W_expr)

    num = sp.expand(num)
    den = sp.expand(den)

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_c = [float(c) for c in num_poly.all_coeffs()]
    den_c = [float(c) for c in den_poly.all_coeffs()]

    tf = signal.TransferFunction(num_c, den_c)

    w = np.logspace(-3, 2, 2000)
    _, mag, _ = signal.bode(tf, w)

    plt.semilogx(w, mag, label=title)




def plot_step(W_expr, title):

    s = sp.symbols('s')

    W_expr = sp.together(W_expr)
    num, den = sp.fraction(W_expr)

    num = sp.expand(num)
    den = sp.expand(den)

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_c = [float(c) for c in num_poly.all_coeffs()]
    den_c = [float(c) for c in den_poly.all_coeffs()]

    tf = signal.TransferFunction(num_c, den_c)

    t, y = signal.step(tf)

    plt.plot(t, y, label=title)




def automatic_lachh_approximation(Wk_expr):

    print("\n=== АВТОМАТИЧНА АПРОКСИМАЦІЯ ЛАЧХ (оптимізація) ===\n")

    s = sp.symbols('s')

    # --- перетворення в числову модель ---
    Wk_expr = sp.together(Wk_expr)
    num, den = sp.fraction(Wk_expr)
    num = sp.expand(num)
    den = sp.expand(den)

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_c = [float(c) for c in num_poly.all_coeffs()]
    den_c = [float(c) for c in den_poly.all_coeffs()]

    tf_exact = signal.TransferFunction(num_c, den_c)

    # частотна сітка
    w = np.logspace(-3, 2, 2000)
    _, mag_exact, _ = signal.bode(tf_exact, w)

    # --- функція похибки ---
    def error_function(params):

        T1, T2, T3, xi = params

        if T1 <= 0 or T2 <= 0 or T3 <= 0 or xi <= 0 or xi >= 2:
            return 1e9

        # формуємо спрощену модель
        Wa = (1/(T1*s)) * \
             (T2**2*s**2 + 2*xi*T2*s + 1) * \
             1/(T3*s + 1)

        Wa = sp.together(Wa)
        n, d = sp.fraction(Wa)
        n = sp.expand(n)
        d = sp.expand(d)

        n_poly = sp.Poly(n, s)
        d_poly = sp.Poly(d, s)

        try:
            n_c = [float(c) for c in n_poly.all_coeffs()]
            d_c = [float(c) for c in d_poly.all_coeffs()]
        except:
            return 1e9

        tf_approx = signal.TransferFunction(n_c, d_c)

        _, mag_approx, _ = signal.bode(tf_approx, w)

        return np.mean((mag_exact - mag_approx)**2)

    # початкові оцінки (можеш міняти)
    initial_guess = [300, 10, 1, 0.7]

    result = minimize(error_function, initial_guess,
                      method='Nelder-Mead',
                      options={'maxiter': 2000})

    T1, T2, T3, xi = result.x

    print("Оптимальні параметри:")
    print(f"T1 = {T1:.4g}")
    print(f"T2 = {T2:.4g}")
    print(f"T3 = {T3:.4g}")
    print(f"ξ  = {xi:.4g}")

    # формуємо фінальну модель
    Wk_approx = (1/(T1*s)) * \
                (T2**2*s**2 + 2*xi*T2*s + 1) * \
                1/(T3*s + 1)

    print("\n--- Спрощена модель ---\n")
    sp.pprint(sp.simplify(Wk_approx), use_unicode=True)

    return Wk_approx

def hybrid_methodical_simplification(Wk_expr):

    print("\n=== ГІБРИДНЕ ПРОФЕСІЙНЕ СПРОЩЕННЯ ===")

    s = sp.symbols('s')

    # ---- Перетворюємо в числову модель ----
    Wk_expr = sp.together(Wk_expr)
    num, den = sp.fraction(Wk_expr)

    num = sp.expand(num)
    den = sp.expand(den)

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_c = np.array([float(c) for c in num_poly.all_coeffs()])
    den_c = np.array([float(c) for c in den_poly.all_coeffs()])

    tf_exact = signal.TransferFunction(num_c, den_c)

    w = np.logspace(-3, 2, 3000)
    _, mag_exact, _ = signal.bode(tf_exact, w)

    # ---- Пошук мінімуму (провалу) ----
    idx_min = np.argmin(mag_exact)
    w_min = w[idx_min]

    window = (w > w_min/10) & (w < w_min*10)
    w_local = w[window]
    mag_local = mag_exact[window]

    logw_local = np.log10(w_local)
    slope_local = np.gradient(mag_local, logw_local)

    idx1 = np.argmin(np.abs(slope_local + 20))
    idx2 = np.argmin(np.abs(slope_local))
    idx3 = np.argmin(np.abs(slope_local - 20))

    w1 = w_local[idx1]
    w2 = w_local[idx2]
    w3 = w_local[idx3]

    print(f"\nАвтоматично знайдено переломи:")
    print(f"ω1 ≈ {w1:.4g}")
    print(f"ω2 ≈ {w2:.4g}")
    print(f"ω3 ≈ {w3:.4g}")

    T1 = 1/w1
    T2 = 1/w2
    T3 = 1/w3

    # ============================================
    # ВИБІР РЕЖИМУ
    # ============================================

    print("\nОберіть режим спрощення:")
    print("1 — Академічний (ξ задається вручну)")
    print("2 — Професійний (оптимізація ξ)")

    mode = input("Ваш вибір (1/2): ")

    # ============================================
    # 1️⃣ АКАДЕМІЧНИЙ РЕЖИМ
    # ============================================

    if mode == "1":

        xi_opt = float(input("Введіть ξ (рекомендовано 0.5–0.95): "))

        print(f"\nПрийнято ξ = {xi_opt}")

    # ============================================
    # 2️⃣ ПРОФЕСІЙНИЙ РЕЖИМ
    # ============================================

    else:

        def error_function(x):
            xi = float(x[0])

            Wa = (1/(T1*s)) * \
                 (T2**2*s**2 + 2*xi*T2*s + 1) / \
                 (T3*s + 1)

            Wa = sp.together(Wa)
            num_a, den_a = sp.fraction(Wa)

            num_a = sp.expand(num_a)
            den_a = sp.expand(den_a)

            num_poly_a = sp.Poly(num_a, s)
            den_poly_a = sp.Poly(den_a, s)

            num_c_a = [float(c) for c in num_poly_a.all_coeffs()]
            den_c_a = [float(c) for c in den_poly_a.all_coeffs()]

            tf_a = signal.TransferFunction(num_c_a, den_c_a)
            _, mag_a, _ = signal.bode(tf_a, w)

            return np.mean((mag_exact - mag_a)**2)

        res = minimize(error_function,
                       x0=[0.7],
                       bounds=[(0.3, 0.99)])

        xi_opt = res.x[0]

        print(f"\nОптимізоване ξ ≈ {xi_opt:.4f}")

    # ============================================
    # ФІНАЛЬНА МОДЕЛЬ
    # ============================================

    Wa_final = (1/(T1*s)) * \
               (T2**2*s**2 + 2*xi_opt*T2*s + 1) / \
               (T3*s + 1)

    print("\n--- Спрощена модель ---\n")
    sp.pprint(sp.simplify(Wa_final), use_unicode=True)

    return Wa_final



def strict_methodical_simplification(Wk_expr):

    print("\n=== СТРОГО МЕТОДИЧНЕ СПРОЩЕННЯ ===")

    s = sp.symbols('s')

    # --- Введення переломів вручну ---
    w1 = float(input("Введіть ω1 (інтегратор): "))
    w2 = float(input("Введіть ω2 (нуль 2-го порядку): "))
    w3 = float(input("Введіть ω3 (полюс 1-го порядку): "))

    T1 = 1 / w1
    T2 = 1 / w2
    T3 = 1 / w3

    print("\nОберіть режим:")
    print("1 — ξ вводиться вручну")
    print("2 — ξ оптимізується")

    mode = input("Ваш вибір (1/2): ")

    # --- Підготовка точної ЛАЧХ ---
    Wk_expr = sp.together(Wk_expr)
    num, den = sp.fraction(Wk_expr)
    num = sp.expand(num)
    den = sp.expand(den)

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_c = [float(c) for c in num_poly.all_coeffs()]
    den_c = [float(c) for c in den_poly.all_coeffs()]

    tf_exact = signal.TransferFunction(num_c, den_c)

    w = np.logspace(-3, 2, 2000)
    _, mag_exact, _ = signal.bode(tf_exact, w)

    # --- Варіант 1: ручний ξ ---
    if mode == "1":

        xi = float(input("Введіть ξ (0.5–0.95): "))
        print(f"\nПрийнято ξ = {xi}")

    # --- Варіант 2: оптимізація ξ ---
    else:

        def error_function(x):
            xi = float(x[0])

            Wa = (1/(T1*s)) * \
                 (T2**2*s**2 + 2*xi*T2*s + 1) / \
                 (T3*s + 1)

            Wa = sp.together(Wa)
            n, d = sp.fraction(Wa)
            n = sp.expand(n)
            d = sp.expand(d)

            n_poly = sp.Poly(n, s)
            d_poly = sp.Poly(d, s)

            n_c = [float(c) for c in n_poly.all_coeffs()]
            d_c = [float(c) for c in d_poly.all_coeffs()]

            tf_a = signal.TransferFunction(n_c, d_c)
            _, mag_a, _ = signal.bode(tf_a, w)

            return np.mean((mag_exact - mag_a)**2)

        res = minimize(error_function, x0=[0.7], bounds=[(0.3, 0.99)])
        xi = res.x[0]

        print(f"\nОптимізоване ξ ≈ {xi:.3f}")

    # --- Формування моделі ---
    Wk_approx = (1/(T1*s)) * \
                (T2**2*s**2 + 2*xi*T2*s + 1) / \
                (T3*s + 1)

    print("\n--- Спрощена модель (методична) ---\n")
    sp.pprint(sp.simplify(Wk_approx), use_unicode=True)

    return Wk_approx


def plot_closed_loop_step(W_expr):

    s = sp.symbols('s')

    W_expr = sp.together(W_expr)
    num, den = sp.fraction(W_expr)

    num = sp.expand(num)
    den = sp.expand(den)

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_c = [float(c) for c in num_poly.all_coeffs()]
    den_c = [float(c) for c in den_poly.all_coeffs()]

    tf = signal.TransferFunction(num_c, den_c)

    t, y = signal.step(tf)

    plt.figure(figsize=(8,5))
    plt.plot(t, y)
    plt.grid(True)
    plt.title("Перехідна характеристика замкненої системи")
    plt.xlabel("t, c")
    plt.ylabel("h(t)")
    plt.show()

    return t, y


def quality_metrics(t, y):

    h_inf = y[-1]          # усталене значення
    h_max = np.max(y)      # максимум

    # Перерегулювання
    sigma = (h_max - h_inf) / h_inf * 100

    # Час регулювання (5%)
    delta = 0.05 * h_inf
    idx = np.where(np.abs(y - h_inf) > delta)[0]

    if len(idx) == 0:
        ts = 0
    else:
        ts = t[idx[-1]]

    print("\n=== Показники якості ===\n")
    print(f"h(∞) ≈ {h_inf:.4f}")
    print(f"h(max) ≈ {h_max:.4f}")
    print(f"Перерегулювання σ ≈ {sigma:.2f} %")
    print(f"Час регулювання ts ≈ {ts:.2f} c")




# ===============================
# main
# ===============================
def main():

    print("=== СИНТЕЗ МЕТОДОМ ЛАЧХ (Нормалізований варіант) ===\n")

    expr_str = input("Введи Wp(s): ")

    expr, num_c, den_c = parse_tf(expr_str)

    tf = signal.TransferFunction(num_c, den_c)

    blocks = decompose_into_blocks(num_c, den_c)



    print("\n=== ДЕКОМПОЗИЦІЯ НА ЛАНКИ ===")

    for i, block in enumerate(blocks, 1):
        kind, val = block
        func_str = block_to_string(block)

        if val is None:
            print(f"W{i}(s): {kind}")
        else:
            print(f"W{i}(s): {func_str}")


    w = np.logspace(-3, 3, 3000)

    # точна ЛАЧХ
    _, mag_exact, _ = signal.bode(tf, w)

    mag_sum = np.zeros_like(w)

    plt.figure(figsize=(9, 6))

    for i, block in enumerate(blocks, 1):
        mag_i = asymptotic_block(block, w)
        mag_sum += mag_i

        label_name = f"W{i}"

        plt.semilogx(w, mag_i, ":", label=label_name)


    plt.semilogx(w, mag_sum, "k--", lw=2,
                 label="Результуюча асимптотична")
    plt.semilogx(w, mag_exact, lw=1.5,
                 label="Точна ЛАЧХ")

    plt.grid(True, which="both")
    plt.xlabel("ω, рад/с")
    plt.ylabel("L(ω), дБ")
    plt.title("Асимптотична та точна ЛАЧХ")
    plt.legend()
    plt.show()

    # ===== Аналіз =====
    print("\n=== АНАЛІЗ ЛАЧХ ===")

    if has_maximum(mag_sum):
        system_type = "max"
        print("ЛАЧХ має максимум → ПРАВА номограма")
    else:
        system_type = "mono"
        print("ЛАЧХ монотонна → ЛІВА номограма")

    tp = float(input("\nВведи час регулювання tp [c]: "))

    if system_type == "max":
        sigma = 6
        w3tp = float(input("\nВведи час регулювання tp [c]: "))
        print(f"Прийнято σm = {sigma} %")
        print(f"Прийнято ω3·tp = {w3tp}")
    else:
        sigma = random.choice([5, 6, 7])
        print(f"Прийнято σm = {sigma} %")
        w3tp = float(input("\nВведи час регулювання tp [c]: "))

    w3 = w3tp / tp
    print(f"\nРозрахована частота зрізу ω3 = {w3:.5g} рад/с")

    # ===== Бажана ЛАЧХ =====
    Lb = build_desired_lachh(w, w3)

    plt.figure(figsize=(8, 5))
    plt.semilogx(w, Lb, "k", lw=2)
    plt.grid(True, which="both")
    plt.xlabel("ω, рад/с")
    plt.ylabel("L_b(ω), дБ")
    plt.title("Бажана логарифмічна АЧХ")
    plt.show()

    # ===== ЛАЧХ КОРИГУЮЧОГО ПРИСТРОЮ =====

    Lk = Lb - mag_sum

    plt.figure(figsize=(9, 6))
    plt.semilogx(w, Lk, "k", lw=2)
    plt.grid(True, which="both")
    plt.xlabel("ω, рад/с")
    plt.ylabel("L_k(ω), дБ")
    plt.title("Асимптотична ЛАЧХ коригуючого пристрою")
    plt.show()

    # ===== ВСІ ЛАЧХ НА ОДНОМУ ГРАФІКУ =====

    Lk = Lb - mag_sum

    plt.figure(figsize=(10, 6))

    plt.semilogx(w, mag_sum, "b--", lw=2, label="Lp(ω) — початкова")
    plt.semilogx(w, Lb, "k", lw=2, label="Lb(ω) — бажана")
    plt.semilogx(w, Lk, "r", lw=2, label="Lk(ω) — коригуючий")

    plt.grid(True, which="both")
    plt.xlabel("ω, рад/с")
    plt.ylabel("L(ω), дБ")
    plt.title("Порівняння ЛАЧХ системи")
    plt.legend()
    plt.show()

    Wk_expr = synthesize_from_breakpoints()

    print("\nОберіть режим спрощення:")
    print("1 — Гібридний професійний")
    print("2 — Строго методичний")

    choice = input("Ваш вибір (1/2): ")

    if choice == "1":
        Wk_approx = hybrid_methodical_simplification(Wk_expr)
    else:
        Wk_approx = strict_methodical_simplification(Wk_expr)


        # 1. Замкнена система
    W_open = sp.simplify(expr * Wk_approx)
    W_closed = sp.simplify(W_open / (1 + W_open))

    print("\n=== Передатна функція замкненої системи з врахуванням корегуючого пристрою ===\n")
    sp.pprint(sp.simplify(W_closed), use_unicode=True)


    # 2. Перехідна
    t, y = plot_closed_loop_step(W_closed)

    # 3. Показники якості
    quality_metrics(t, y)




if __name__ == "__main__":
    main()
