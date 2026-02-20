
pkg load control
graphics_toolkit("gnuplot");
setenv("GNUTERM","png");

num = [100.0, 880.0000000000001, 1679.9999999999998];
den = [13.0, 180.0, 893.0000000000001, 1685.9999999999998];

W = tf(num, den);

t = linspace(0, 200, 20000);

% ==================================================
% 1) ЛАПЛАС-ПОДІБНИЙ МЕТОД (residue)
% ==================================================
den_step = conv(den, [1 0]);
[r, p, k] = residue(num, den_step);

y_laplace = zeros(size(t));
for i = 1:length(r)
    y_laplace += real(r(i) * exp(p(i) * t));
end
y_laplace = y_laplace / y_laplace(end);

figure;
plot(t, y_laplace, "k", "linewidth", 1.5);
grid on;
xlabel("t, c");
ylabel("h(t)");
title("Перехідна характеристика (Laplace)");
print("transition_laplace.png", "-dpng");

% ==================================================
% 2) ЧИСЕЛЬНИЙ МЕТОД (step)
% ==================================================
[y_step, t_step] = step(W, t);

figure;
plot(t_step, y_step, "k", "linewidth", 1.5);
grid on;
xlabel("t, c");
ylabel("y(t)");
title("Перехідна характеристика (step)");
print("transition_step.png", "-dpng");
