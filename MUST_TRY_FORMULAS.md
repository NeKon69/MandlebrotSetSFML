1. Burning ship - there is nothing special about it, usual thing you see when talking about fractals:
```
new_real = (z_real * z_real - z_imag * z_imag) + real;
z_imag = T(2.0) * fabs(z_real) * fabs(z_imag) + imag;
```
2. Celtic Heart - that one looks cool and is not that hard to implement:
```
new_real = fabs(z_real * z_real - z_imag * z_imag) + real;
z_imag = T(-2.0) * z_real * z_imag + imag;
```
3. Cubed Mandelbrot - that one is a bit more complex, but still doable:
```
T zr_sq = z_real * z_real;
T zi_sq = z_imag * z_imag;
new_real = z_real * (zr_sq - T(3.0) * zi_sq) + real;
z_imag = z_imag * (T(3.0) * zr_sq - zi_sq) + imag;
```
4. Exponential Mandelbrot - it's like a half sphere with some twists:
```
T exp_zr = exp(z_real);
new_real = exp_zr * cos(z_imag) + real;
z_imag = exp_zr * sin(z_imag) + imag;
```
5. Sine Mandelbrot - trigonometric functions are ass slow on gpu, but with them, fractals look great:
```
new_real = sin(z_real) * cosh(z_imag) + real;
z_imag = cos(z_real) * sinh(z_imag) + imag;
```
6. Buffalo fractal - looks like a goddam space ship idk:
```
new_real = z_real*z_real - z_imag*z_imag - z_real + real;
z_imag = T(2.0)*z_real*z_imag + z_imag + imag;
```
7. Fish Fractal - just swap z_imag and z_real and you get a goddam boomerang:
```
new_real = z_imag*z_imag - z_real*z_real + real;
z_imag = T(2.0)*z_real*z_imag + imag;
```
8. Drill Fractal - add some multiplication by square root of PI and you get something referring to a drill, looks sick!
```
new_real = (z_real * z_real - z_imag * z_imag) * sqrt(PI) + (real);
z_imag = T(2.0) * z_real * z_imag + imag;
```
9. 3D Fractal - if you look closely at julia with correct palette, you might think you are in 3d: just mouse over near the black circle! (suggestion: try "Oscillating Grayscale):
```
new_real = z_real * z_comp + real;
z_imag = z_imag * z_comp + imag;
```
10. Field of contused fractals - it almost as if you just did a lot of math and got a lot of repeating fractals:
```
T temp_real = z_real * z_real - z_imag * z_imag + real;
T temp_imag = T(2.0) * z_real * z_imag + imag;
new_real = tan(temp_real);
z_imag = tan(temp_imag);
```
11. Singularity fractal - idk what should I even call it, just look at the julia set, it looks sick:
```
T zr_sq = z_real * z_real;
T zi_sq = z_imag * z_imag;
new_real = zr_sq - zi_sq + real;
z_imag = T(2.0) * real * z_imag + imag;
```
12. Crazybrot - if you zoom in the center you can hover a mouse for a bit and get some "balls" - literally!
```
T inv_z_comp = T(1.0) / (z_comp + T(1e-9));
new_real = z_real * inv_z_comp + real;
z_imag = -z_imag * inv_z_comp + imag;
```
13. Damaged DoubleBrot - just a bit of damage to the doubled mandelbrot fractal and all because of GOLDEN_RATIO:
```
T mag = sqrt(z_comp + T(1e-9));
T angle = atan2(z_imag, z_real);
T mag_pow_gr = pow(mag, T(GOLDEN_RATIO));
T angle_gr = GOLDEN_RATIO * angle;
new_real = mag_pow_gr * cos(angle_gr) + real;
z_imag = mag_pow_gr * sin(angle_gr) + imag;
```
14. LightningBrot - julia may look like a lightning bolt in some places (again, try "Oscillating Grayscale" palette):
```
T angle = atan2(z_imag, z_real);
T cos_angle = cos(angle);
new_real = (z_real * z_real - z_imag * z_imag) + real * cos_angle;
z_imag = T(2.0) * z_real * z_imag + imag * cos_angle;
```
15. Prison fractal - looks like a prison on julia side:
```
T mod_real = sin(z_real);
T mod_imag = sin(z_imag);
new_real = (mod_real * mod_real - mod_imag * mod_imag) + real;
z_imag = T(2.0) * mod_real * mod_imag + imag;
```
16. Eaten fractal - this one is the most normal in terms of looking, it has nice symmetry, just looks like it was eaten or it's sick or smth idk:
```
T zr2 = z_real * z_real;
T zi2 = z_imag * z_imag;
T a = zr2 - zi2;
T b = T(2.0) * z_real * z_imag;
T den_real = a + T(0.1);
T den_imag = b;
T den_mag_sq = den_real * den_real + den_imag * den_imag + T(1e-9);
T div_real = (real * den_real + imag * den_imag) / den_mag_sq;
T div_imag = (imag * den_real - real * den_imag) / den_mag_sq;
new_real = a + div_real;
z_imag = b + div_imag;
```
17. Newton fractal - if you pick exact center at the mandelbrot you can see the newton fractal:
```
T zr2 = z_real * z_real;
T zi2 = z_imag * z_imag;
T a = zr2 - zi2;
T b = T(2.0) * z_real * z_imag;
T p = z_real * a - z_imag * b;
T q = z_real * b + z_imag * a;
T num_real = p - T(1.0);
T num_imag = q;
T den_real = T(3.0) * a;
T den_imag = T(3.0) * b;
T den_mag_sq = den_real * den_real + den_imag * den_imag + T(1e-9);
T div_real = (num_real * den_real + num_imag * den_imag) / den_mag_sq;
T div_imag = (num_imag * den_real - num_real * den_imag) / den_mag_sq;
new_real = z_real - div_real + real;
z_imag = z_imag - div_imag + imag;
```
18. Undefined fractal - this one is just a bit of a mess, with white parts being undefined (division by zero or smth):
```
T w_real = (z_real * z_real - z_imag * z_imag) + real;
T w_imag = T(2.0) * z_real * z_imag + imag;
new_real = cos(w_real) * cosh(w_imag);
z_imag = -sin(w_real) * sinh(w_imag);
```
19. Space Fractal - with right palette and zoom you can see space:
```
T den_real = z_real + T(0.01);
T den_imag = z_imag;
T den_mag_sq = den_real * den_real + den_imag * den_imag + T(1e-9);
T div_real = (real * den_real + imag * den_imag) / den_mag_sq;
T div_imag = (imag * den_real - real * den_imag) / den_mag_sq;
new_real = (z_real * z_real - z_imag * z_imag) + div_real;
z_imag = T(2.0) * z_real * z_imag + div_imag;
```
20. Washed away fractal - looks like a washed away mandelbrot with following twists:
```
T zr2 = z_real * z_real;
T zi2 = z_imag * z_imag;
T a = zr2 - zi2;
T b = T(2.0) * z_real * z_imag;
T log_mag = T(0.5) * log(z_comp + T(1e-9));
T cos_rot = cos(log_mag);
T sin_rot = sin(log_mag);
T rotated_a = a * cos_rot - b * sin_rot;
T rotated_b = a * sin_rot + b * cos_rot;
new_real = rotated_a + real;
z_imag = rotated_b + imag;
```
21. Angrybrot - it's just a mandelbrot but more aggressive and sharp:
```
T atan_zr = atan(z_real);
T atan_zi = atan(z_imag);
new_real = (z_real * z_real - atan_zi * atan_zi) + real;
z_imag = T(2.0) * atan_zr * z_imag + imag;
```
22. Car Road - the julia set looks lire a road on correct mouse position:
```
T sinh_zr = sinh(z_real);
T cosh_zr = cosh(z_real);
T sin_zi = sin(z_imag);
T cos_zi = cos(z_imag);
T base_real = sinh_zr * cos_zi;
T base_imag = cosh_zr * sin_zi;
T angle_rot = sqrt(z_comp + T(1e-9));
T cos_rot = cos(angle_rot);
T sin_rot = sin(angle_rot);
T rotated_real = base_real * cos_rot - base_imag * sin_rot;
T rotated_imag = base_real * sin_rot + base_imag * cos_rot;
new_real = rotated_real + real;
z_imag = rotated_imag + imag;
```
23. Polar cowlick fractal - julia looks like a cowlick on the edges of mandelbrot:
```
T r = sqrt(z_comp + T(1e-9));
T theta = atan2(z_imag, z_real);
T r_new = sin(r * T(3.0));
T theta_new = theta + r;
new_real = r_new * cos(theta_new) + real;
z_imag = r_new * sin(theta_new) + imag;
```
24. Bullet shot fractal - looks like a bullets shot through fractal (from both sides) and fractal stopped it:
```
T w1_real = z_real * real - z_imag * imag;
T w1_imag = z_real * imag + z_imag * real;
T sin_w1_real = sin(w1_real) * cosh(w1_imag);
T sin_w1_imag = cos(w1_real) * sinh(w1_imag);
T w2_real = z_real + real;
T w2_imag = z_imag + imag;
T cos_w2_real = cos(w2_real) * cosh(w2_imag);
T cos_w2_imag = -sin(w2_real) * sinh(w2_imag);
new_real = (z_real * z_real - z_imag * z_imag) + sin_w1_real + cos_w2_real;
z_imag = (T(2.0) * z_real * z_imag) + sin_w1_imag + cos_w2_imag;
```
25. Vase with 3d effect - looks like a vase being filled up with water + some 3d effects on specific parts:
```
T log_real = T(0.5) * log(z_comp + T(1e-9));
T log_imag = atan2(z_imag, z_real);
T M_real = log_real - log_imag;
T M_imag = log_real + log_imag;
T exp_M_real = exp(M_real);
new_real = exp_M_real * cos(M_imag) + real;
z_imag = exp_M_real * sin(M_imag) + imag;
```
New formulas are coming soon, stay tuned!