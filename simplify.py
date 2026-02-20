import sympy as sp

def normalize_linear(factor, s):
    factor = sp.expand(factor)
    a = factor.coeff(s, 1)
    b = factor.coeff(s, 0)

    if b != 0:
        T = a / b
        return b, (T*s + 1)
    return 1, factor


def decompose_transfer_function():
    s = sp.symbols('s')
    
    user_input = input("Введіть передавальну функцію W(s):\n")
    W = sp.sympify(user_input)
    
    numerator, denominator = sp.fraction(sp.simplify(W))
    
    num_coeff, num_factors = sp.factor_list(numerator)
    den_coeff, den_factors = sp.factor_list(denominator)
    
    # Методичний варіант
    K = num_coeff
    
    links = []
    
    # --- Чисельник ---
    for factor, power in num_factors:
        for _ in range(power):
            if sp.degree(factor) == 1:
                coef, norm = normalize_linear(factor, s)
                K *= coef
                links.append(norm)
            else:
                links.append(factor)
    
    # --- Знаменник ---
    first_quad = True

    for factor, power in den_factors:
        for _ in range(power):
            if sp.degree(factor) == 1:
                coef, norm = normalize_linear(factor, s)
                K /= coef
                links.append(1/norm)
            elif sp.degree(factor) == 2:
                if first_quad:
                    factor = den_coeff * factor
                    first_quad = False
                links.append(1/factor)
            else:
                links.append(1/factor)

    # 🔥 Ось цього блоку у тебе не було
    print("\nРезультат декомпозиції:\n")
    print("K =", sp.simplify(K))

    for i, link in enumerate(links, 1):
        print(f"W{i}(s) =", sp.simplify(link))


if __name__ == "__main__":
    decompose_transfer_function()
