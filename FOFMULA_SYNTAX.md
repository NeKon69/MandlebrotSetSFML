# Custom Fractal Formulas (NVRTC)

This fractal explorer allows you to define your own fractal iteration logic using a simple C++-like syntax that is compiled and run directly on your GPU using NVRTC (NVIDIA Runtime Compilation). This gives you incredible flexibility to explore new shapes beyond the standard Mandelbrot or Julia sets.

You don't need to know CUDA programming or kernel details. You just need to understand the variables available to you and how to write mathematical expressions to update them.

## How to Use Custom Formulas

You will find a specific input area (likely a text box) within the application where you can type your custom formula code.

Your code snippet represents the core **iteration step** for a single point in the complex plane. This code will be executed repeatedly for each pixel, up to the maximum number of iterations you've set.

## Available Variables

Inside your formula code, you can use the following predefined variables. These represent the current state of the iteration and the initial point being tested.

*   `z_real`: The real part of the currently iterating complex number, *z*.
*   `z_imag`: The imaginary part of the currently iterating complex number, *z*.
*   `real`: The real part of the initial complex number, *c* (the point corresponding to the pixel being calculated). This value is constant for a given pixel throughout its iterations.
*   `imag`: The imaginary part of the initial complex number, *c* (the point corresponding to the pixel being calculated). This value is constant for a given pixel throughout its iterations.
*   `new_real`: **Important:** A temporary variable where you should store the *calculated next real part* of *z*. After your code snippet finishes, the system will automatically update `z_real = new_real;` for the next iteration.
*   `z_comp`: The squared magnitude of the current complex number *z*, calculated as `z_real * z_real + z_imag * z_imag`. This is used by the system to check the escape condition (usually `z_comp < 4`), but you can also use it in your own calculations.
*   `PI`: The mathematical constant π (approximately 3.14159...).
*   `GOLDEN_RATIO`: The mathematical constant φ (the Golden Ratio, approximately 1.618...).

All these variables (`z_real`, `z_imag`, `real`, `imag`, `new_real`, `z_comp`, `PI`, `GOLDEN_RATIO`) will be of the same floating-point type (`float` or `double`) that you've selected for rendering precision in the application.

## Writing Your Formula

Your formula code should define how the complex number *z* evolves in each iteration. This is typically done by calculating the *next* values for `z_real` and `z_imag` based on their current values (`z_real`, `z_imag`) and the initial point (`real`, `imag`).

Your code should contain C++-like assignment statements that update the `new_real` and `z_imag` variables.

**Syntax Rules:**

1.  Follow basic C++ syntax. Each statement must end with a semicolon `;`.
2.  Use the variables listed above.
3.  Use standard arithmetic operators: `+`, `-`, `*`, `/`.
4.  You can use parentheses `()` for grouping operations.
5.  You can call available mathematical functions (see below).

**Example: The Standard Mandelbrot Formula**

The standard Mandelbrot set is defined by the iteration `z = z^2 + c`. In terms of real and imaginary parts (`z = z_real + i * z_imag`, `c = real + i * imag`), this expands to:

$(z_{real} + i z_{imag})^2 + (real + i imag)$
$= (z_{real}^2 - z_{imag}^2 + 2 i z_{real} z_{imag}) + (real + i imag)$
$= (z_{real}^2 - z_{imag}^2 + real) + i (2 z_{real} z_{imag} + imag)$

So, the next real part is $(z_{real}^2 - z_{imag}^2 + real)$ and the next imaginary part is $(2 z_{real} z_{imag} + imag)$.

In your custom formula input, you would write this as:

```cpp
// Calculate the next real part and store it in new_real
new_real = (z_real * z_real - z_imag * z_imag) + real;

// Calculate the next imaginary part and update z_imag directly
z_imag = 2 * z_real * z_imag + imag;

// The system automatically performs: z_real = new_real; after your code.
```

**Another Example: Burning Ship Fractal**

The Burning Ship fractal uses the iteration `z = (|real(z)| + i |imag(z)|)^2 + c`.

**Burning Ship Example:**

```cpp
// Calculate the next real part and store it in new_real
new_real = (z_real * z_real - z_imag * z_imag) + real;

// Calculate the next imaginary part and update z_imag directly
z_imag = 2 * fabs(z_real) * fabs(z_imag) + imag; // Note the use of fabs()

// The system automatically performs: z_real = new_real; after your code.
```

Remember, your code only defines the update rules for `new_real` and `z_imag`. The loop, escape condition (`z_comp < 4`), and iteration count are handled automatically by the system.

## Available Mathematical Functions

You can use many common mathematical functions from the CUDA math library within your formula. These function names are the same as in standard C/C++.

**Standard Precision Functions:**
These functions provide higher precision and generally correspond to the standard C math functions. The actual precision depends on whether the explorer is rendering in `float` or `double` mode.

Examples:
*   `sin(x)`
*   `cos(x)`
*   `tan(x)`
*   `asin(x)`
*   `acos(x)`
*   `atan(x)`
*   `sinh(x)`
*   `cosh(x)`
*   `tanh(x)`
*   `sqrt(x)`
*   `cbrt(x)` (Cube root)
*   `exp(x)` ($e^x$)
*   `log(x)` (Natural logarithm)
*   `log10(x)` (Base-10 logarithm)
*   `pow(base, exponent)` (Base raised to the power of exponent)
*   `fabs(x)` (Absolute value of a floating-point number)
*   `fmax(x, y)` (Maximum of two numbers)
*   `fmin(x, y)` (Minimum of two numbers)
*   `fmod(x, y)` (Floating-point remainder of x/y)

And many others. If a function exists in the standard C math library (like `sin`, `cos`, `sqrt`, `log`, `exp`, `fabs`, etc.), it's likely available.

**Faster, Lower Precision Functions (Suffixed with `f`):**
For single-precision (`float`) rendering, you can often use faster, lower-precision versions of common functions. These typically have names ending in `f`. Using these might slightly reduce visual quality but can significantly increase rendering speed, especially with `float` precision.

Examples:
*   `__sinf(x)` (Faster sine for float)
*   `__cosf(x)` (Faster cosine for float)
*   `__tanf(x)` (Faster tangent for float)
*   `__sqrtf(x)` (Faster square root for float)
*   `__expf(x)` (Faster $e^x$ for float)
*   `__logf(x)` (Faster natural logarithm for float)
*   `__powf(base, exponent)` (Faster power for float)
*   `__fabsf(x)` (Faster absolute value for float - same as `fabsf`)
*   `__fmaxf(x, y)` (Faster maximum for float - same as `fmaxf`)
*   `__fminf(x, y)` (Faster minimum for float - same as `fminf`)

Note: Using the `__f` variants when rendering in `double` precision will likely result in compilation errors. Stick to the standard functions when using `double` precision.

## Using Hardcoded Numbers (Literals)

If you need to use a specific numerical constant in your formula (like `0.5`, `2.0`, `-1.7`), it's crucial to ensure it matches the floating-point precision currently used by the renderer (`float` or `double`).

**Always wrap your hardcoded numbers with `T()`**. `T` is a placeholder that will be replaced by the correct precision type (`float` or `double`) during compilation.

*   **Correct:** `new_real = z_real + T(0.5);`
*   **Incorrect (Potential Issues):** `new_real = z_real + 0.5;` (This `0.5` might be treated as a `double` literal even in `float` mode, potentially causing unexpected behavior or performance issues).

So, when writing numbers directly into your formula, write them like `T(1.0)`, `T(2.5)`, `T(-0.1)`, `T(1e-5)`, etc.

## Summary & Tips

*   Your custom code defines how `z_real` and `z_imag` change in *each* iteration.
*   You must calculate the *next* real part and assign it to `new_real`.
*   You must calculate the *next* imaginary part and assign it to `z_imag`.
*   The system updates `z_real = new_real;` automatically after your code runs.
*   Use the available variables: `z_real`, `z_imag`, `real`, `imag`, `new_real`, `z_comp`, `PI`, `GOLDEN_RATIO`.
*   End each statement with a semicolon `;`.
*   Use `T(...)` for all hardcoded numbers (e.g., `T(2.0)`, `T(-0.75)`).
*   You can use many standard math functions (`sin`, `cos`, `sqrt`, `fabs`, `pow`, etc.).
*   For `float` precision, consider using faster versions like `__sinf`, `__sqrtf`, `__fabsf`.
*   Complex formulas might take longer to render.

Have fun exploring the endless possibilities of fractal generation!
