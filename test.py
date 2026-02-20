import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import brentq

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

s = sp.symbols('s')

# ===============================
# Парсинг TF
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

    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]

    return expr, signal.TransferFunction(num_coeffs, den_coeffs)


# ===============================
# Форматування TF
# ===============================
def tf_to_string(tf, var="s"):
    def poly_str(coeffs):
        deg = len(coeffs) - 1
        terms = []

        for i, c in enumerate(coeffs):
            p = deg - i
            if abs(c) < 1e-12:
                continue

            c = round(c, 3)

            if p == 0:
                term = f"{c}"
            elif p == 1:
                term = f"{c}{var}" if c != 1 else f"{var}"
            else:
                term = f"{c}{var}^{p}" if c != 1 else f"{var}^{p}"

            terms.append(term)

        return " + ".join(terms).replace("+ -", "- ")

    return f"\n      {poly_str(tf.num)}\nW(s)= -----------------------------\n      {poly_str(tf.den)}"



# ===============================
# Системи
# ===============================
def open_loop(plant, ctrl):
    return signal.TransferFunction(
        np.polymul(plant.num, ctrl.num),
        np.polymul(plant.den, ctrl.den)
    )


def closed_loop(Wp):
    return signal.TransferFunction(
        Wp.num,
        np.polyadd(Wp.den, Wp.num)
    )


# ===============================
# Стійкість (кореневий)
# ===============================
def check_stability_root(tf):
    poles = np.roots(tf.den)
    stable = np.all(np.real(poles) < 0)
    return stable, poles


def explain_frequency_stability(open_loop_tf):

    print("\n=========================== Стійкість (Частотний метод) ===========================")

    # ---------------------------------------------------------
    # 0. Праві полюси розімкненої системи
    # ---------------------------------------------------------

    poles_open = np.roots(open_loop_tf.den)
    P = np.sum(np.real(poles_open) > 0)

    print(f"\n0) Кількість правих полюсів розімкненої системи P = {P}")

    # ---------------------------------------------------------
    # 1. Побудова частотних характеристик
    # ---------------------------------------------------------

    w = np.logspace(-5, 5, 60000)
    w, mag_db, phase_deg = signal.bode(open_loop_tf, w=w)
    mag = 10**(mag_db / 20)

    # ---------------------------------------------------------
    # 2. Запаси стійкості (тільки якщо P = 0)
    # ---------------------------------------------------------

    if P == 0:
        print("\n1) Обчислення запасів стійкості (P = 0)")

        # --- Частота зрізу ---
        def f_mag(omega):
            _, m_db, _ = signal.bode(open_loop_tf, w=[omega])
            return 10**(m_db[0]/20) - 1

        wc = None
        for i in range(len(w)-1):
            if (mag[i]-1)*(mag[i+1]-1) < 0:
                try:
                    wc = brentq(f_mag, w[i], w[i+1])
                except ValueError:
                    wc = None
                break

        if wc:
            _, m_db_c, phase_c = signal.bode(open_loop_tf, w=[wc])
            PM = 180 + phase_c[0]

            print(f"\n   ωc = {wc:.6f} рад/с")
            print(f"   φ(ωc) = {phase_c[0]:.6f}°")
            print(f"   PM = {PM:.6f}°")
        else:
            PM = None
            print("   Частота зрізу не знайдена.")

        # --- Частота -180 ---
        def f_phase(omega):
            _, _, ph = signal.bode(open_loop_tf, w=[omega])
            return ph[0] + 180

        w180 = None
        for i in range(len(w)-1):
            if (phase_deg[i]+180)*(phase_deg[i+1]+180) < 0:
                try:
                    w180 = brentq(f_phase, w[i], w[i+1])
                except ValueError:
                    w180 = None
                break

        if w180:
            _, m_db_180, _ = signal.bode(open_loop_tf, w=[w180])
            mag_180 = 10**(m_db_180[0]/20)
            GM = 1/mag_180
            GM_db = -m_db_180[0]

            print(f"\n   ω180 = {w180:.6f} рад/с")
            print(f"   GM = {GM:.6f}")
            print(f"   GM(dB) = {GM_db:.6f} dB")
        else:
            GM = np.inf
            print("\n   Фаза не досягає -180° → GM = ∞")

    else:
        print("\n1) P ≠ 0 → Запаси стійкості не застосовуються.")
        PM = None
        GM = None

    # ---------------------------------------------------------
    # 3. Повний критерій Найквіста
    # ---------------------------------------------------------

    print("\n2) Аналіз за повним критерієм Найквіста")

    w_nyq = np.logspace(-5, 5, 80000)
    w_nyq, H = signal.freqresp(open_loop_tf, w_nyq)

    angles = np.unwrap(np.angle(H + 1))
    total_angle = angles[-1] - angles[0]
    N = int(round(total_angle / (2*np.pi)))

    Z = N + P

    print(f"   N (охоплення точки -1,0) = {N}")
    print(f"   Z = N + P = {Z}")

    if Z == 0:
        print("   Замкнена система СТІЙКА.")
    else:
        print("   Замкнена система НЕСТІЙКА.")

    # ---------------------------------------------------------
    # 4. Побудова графіків
    # ---------------------------------------------------------

    # Найквіст
    plt.figure()
    plt.plot(H.real, H.imag)
    plt.plot(H.real, -H.imag)
    plt.scatter(-1, 0)
    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Діаграма Найквіста")
    plt.grid(True)
    plt.savefig("Діаграма_Нейквіста.png", dpi=300)
    plt.close()

    print("\n✔ Збережено графіки:")
    print(" - ЛАЧХ.png")
    print(" - ЛФЧХ.png")
    print(" - Діаграма_Нейквіста.png")


def plot_bode_system(tf, name_prefix):
    """
    Побудова ЛАЧХ та ЛФЧХ для будь-якої системи
    """

    w = np.logspace(-5, 5, 60000)
    w, mag_db, phase_deg = signal.bode(tf, w=w)

    # ----- ЛАЧХ -----
    plt.figure()
    plt.semilogx(w, mag_db)
    plt.grid(True, which="both")
    plt.xlabel("ω, рад/с")
    plt.ylabel("L(ω), dB")
    plt.title(f"ЛАЧХ {name_prefix}")
    plt.savefig(f"{name_prefix}_ЛАЧХ.png", dpi=300)
    plt.close()

    # ----- ЛФЧХ -----
    plt.figure()
    plt.semilogx(w, phase_deg)
    plt.grid(True, which="both")
    plt.xlabel("ω, рад/с")
    plt.ylabel("φ(ω), град")
    plt.title(f"ЛФЧХ {name_prefix}")
    plt.savefig(f"{name_prefix}_ЛФЧХ.png", dpi=300)
    plt.close()

    print(f"✔ Побудовано Графік для {name_prefix}")




# ===============================
# Масштаб
# ===============================
def ask_plot_scale():
    print("\n--- Масштаб графіків ---")
    t_end = input("Кінцевий час t_end (Enter = авто): ")
    t_end = float(t_end) if t_end.strip() else None

    use_ylim = input("Обмежити Y? (y/n, Enter = ні): ").lower()
    if use_ylim == "y":
        y1 = float(input("y_min = "))
        y2 = float(input("y_max = "))
        ylim = (y1, y2)
    else:
        ylim = None

    return t_end, ylim


# ===============================
# Чисельний step
# ===============================
def plot_step_numeric(sys, title, fname, t_end, dt, ylim):
    t = np.arange(0, t_end + dt, dt)
    _, y = signal.step(sys, T=t)

    plt.figure()
    plt.plot(t, y, lw=1.4)
    plt.grid(True)
    plt.xlim(0, t_end)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel("t, c")
    plt.ylabel("y(t)")
    plt.title(title)
    plt.savefig(fname, dpi=300)
    plt.close()


# ===============================
# Laplace-подібний
# ===============================
def plot_step_laplace(sys, title, fname, t_end, dt, ylim):
    num, den = sys.num, sys.den
    den_s = np.convolve(den, [1, 0])
    r, p, _ = signal.residue(num, den_s)

    t = np.arange(0, t_end + dt, dt)
    y = np.zeros_like(t)

    for i in range(len(r)):
        y += np.real(r[i] * np.exp(p[i] * t))

    if abs(y[-1]) > 1e-9:
        y /= y[-1]

    plt.figure()
    plt.plot(t, y, "k", lw=1.2)
    plt.grid(True)
    plt.xlim(0, t_end)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel("t, c")
    plt.ylabel("h(t)")
    plt.title(title)
    plt.savefig(fname, dpi=300)
    plt.close()


def quality_from_graph_inputs():

    print("\n================ ВИЗНАЧЕННЯ ПОКАЗНИКІВ ЯКОСТІ =================")
    print("Значення необхідно зчитати безпосередньо з перехідної характеристики.\n")

    print("ЩО ВВОДИТИ:")
    print("h_max  – максимальне значення першого піку")
    print("h_уст  – усталене значення (рівень, до якого сходиться графік)")
    print("h1     – амплітуда першого коливання (відносно h_уст)")
    print("h2     – амплітуда другого коливання (відносно h_уст)")
    print("t1     – час першого максимуму")
    print("t2     – час другого максимуму")
    print("tp     – час регулювання (вхід у смугу ±5%)")
    print("N      – кількість коливань до входу в режим\n")

    # ----- Ввід -----
    h_max = float(input("h_max = "))
    h_ust = float(input("h_уст = "))
    h1 = float(input("h1 = "))
    h2 = float(input("h2 = "))
    t1 = float(input("t1 [c] = "))
    t2 = float(input("t2 [c] = "))
    tp = float(input("tp [c] = "))
    N = int(input("N = "))

    print("\n================ РОЗРАХУНКИ =================")

    # ----- Перерегулювання -----
    if h_ust == 0:
        print("ПОМИЛКА: h_уст не може дорівнювати 0.")
        return

    sigma = (h_max - h_ust) / h_ust * 100

    print("\n1) Максимальне перерегулювання:")
    print("σ = (h_max − h_уст) / h_уст · 100%")
    print(f"σ = ({h_max} − {h_ust}) / {h_ust} · 100%")
    print(f"σ = {sigma:.3f} %")

    # ----- Період коливань -----
    if t2 <= t1:
        print("\nПОМИЛКА: t2 повинно бути більше t1.")
        T_k = None
    else:
        T_k = t2 - t1
        print("\n2) Період коливань:")
        print("T_k = t2 − t1")
        print(f"T_k = {t2} − {t1} = {T_k:.3f} c")

    # ----- Декремент -----
    if h2 == 0:
        print("\nПОМИЛКА: h2 не може дорівнювати 0.")
        chi = None
    else:
        chi = h1 / h2
        print("\n3) Декремент згасання:")
        print("χ = h1 / h2")
        print(f"χ = {h1} / {h2} = {chi:.3f}")

    # ----- Статична похибка -----
    eps = abs(1 - h_ust)

    print("\n4) Статична похибка:")
    print("ε = |1 − h_уст|")
    print(f"ε = |1 − {h_ust}| = {eps:.6f}")

    print("\n5) Час регулювання:")
    print(f"tp = {tp} c")

    print("\n6) Кількість коливань:")
    print(f"N = {N}")

    # ----- Перевірка умов -----
    print("\n================ ПЕРЕВІРКА УМОВ =================")

    print("\nПеревірка σ ≤ 10%:")
    if sigma <= 10:
        print(f"{sigma:.3f}% ≤ 10% → Умова виконується ✔")
    else:
        print(f"{sigma:.3f}% > 10% → Умова НЕ виконується ✖")

    print("\nПеревірка ε = 0:")
    if eps < 1e-3:
        print(f"{eps:.6f} ≈ 0 → Умова виконується ✔")
    else:
        print(f"{eps:.6f} ≠ 0 → Умова НЕ виконується ✖")

    print("\n===================================================")

    return {
        "sigma": sigma,
        "T_k": T_k,
        "chi": chi,
        "eps": eps,
        "tp": tp,
        "N": N
    }



# ===============================
# main
# ===============================
def main():
    print("=========================== Введіть значення передатніх функцій ===========================")
    print("Приклади для тестування:")
    print("Wp(s): s+1               | Wk(s): 125/((s+0.3)*(0.03s^2+0.07s+1))")
    print("Wp(s): (s+0.16)/(s+0.01) | Wk(s): 0.25/(150s^2+5s+1)")
    print("Wp(s): s+0.12            | Wk(s): 4/((s+0.4)*(0.1s^2+0.03s+1))")

    wp_str = input("\nВведи Wp(s): ")
    wk_str = input("Введи Wk(s): ")

    wp_expr, wp_tf = parse_tf(wp_str)
    wk_expr, wk_tf = parse_tf(wk_str)

    Wp = open_loop(wp_tf, wk_tf)
    W  = closed_loop(Wp)

    print("\n=========================== Отримані системи ===========================")
    Wp_expr_full = sp.simplify(wp_expr * wk_expr)
    W_expr_full = sp.simplify(Wp_expr_full / (1 + Wp_expr_full))

    print("\nПередатна функція розімкненої системи Wp(s):", Wp_expr_full)
    print("\nПередатна функція замкненої системи W(s) =", W_expr_full)


    # ---- Кореневий метод ----
    root_stable, poles = check_stability_root(W)

    print("\n=========================== Стійкість (Кореневий метод) ===========================")
    for p in poles:
        print(f"s = {p:.6f}")
    print("СТІЙКА" if root_stable else "НЕСТІЙКА")

    # ---- Частотний метод ----
    explain_frequency_stability(Wp)


    # --- ЛАЧХ та ЛФЧХ розімкненої ---
    plot_bode_system(Wp, "Розімкнена система")

    # --- ЛАЧХ та ЛФЧХ замкненої ---
    plot_bode_system(W, "Замкнена система")



    # ---- Масштаб ----
    t_user, ylim = ask_plot_scale()
    t_num = t_user if t_user else 20
    t_lap = t_user if t_user else 200

    # ---- Графіки ----
    plot_step_numeric(
        W,
        "Перехідна характеристика САУ (чисельний метод)",
        "step_numeric.png",
        t_num, 0.001, ylim
    )

    plot_step_laplace(
        W,
        "Перехідна характеристика САУ (Laplace-подібний метод)",
        "step_laplace.png",
        t_lap, 0.01, ylim
    )

    use_manual = input(
    "\nВвести показники якості за графіком і виконати розрахунок? (y/n): "
    ).lower()

    if use_manual == "y":
        quality_from_graph_inputs()



if __name__ == "__main__":
    main()