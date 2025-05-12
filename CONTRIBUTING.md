# Contributing to CUDA Fractal Explorer

First off, thank you for considering contributing to CUDA Fractal Explorer! 🎉 We appreciate your interest in helping make this project better. Whether it's reporting a bug, discussing features, or writing code, your contributions are welcome.

This document provides guidelines for contributing to the project. Please read it carefully to ensure a smooth contribution process.

## Table of Contents

*   [Code of Conduct](#code-of-conduct)
*   [How Can I Contribute?](#how-can-i-contribute)
    *   [Reporting Bugs](#reporting-bugs)
    *   [Suggesting Enhancements](#suggesting-enhancements)
    *   [Your First Code Contribution](#your-first-code-contribution)
    *   [Pull Requests](#pull-requests)
*   [Development Setup](#development-setup)
*   [Style Guide](#style-guide)
*   [Questions?](#questions)

## Code of Conduct

This project and everyone participating in it are governed by a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [nobodqwe@gmail.com](mailto:nobodqwe@gmail.com).

## How Can I Contribute?

There are many ways to contribute:

### Reporting Bugs

If you find a bug, please ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/NeKon69/MandlebrotSetSFML/issues).

If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/NeKon69/MandlebrotSetSFML/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample or an executable test case** demonstrating the expected behavior that is not occurring.

Please include:

*   Your operating system name and version.
*   Your NVIDIA GPU model and driver version.
*   The CUDA Toolkit version you are using.
*   Your C++ compiler name and version.
*   Steps to reproduce the bug.
*   What you expected to happen.
*   What actually happened.
*   Screenshots or logs if applicable.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one:

1.  Check the [Issues](https://github.com/NeKon69/MandlebrotSetSFML/issues) list to see if the enhancement has already been suggested.
2.  If not, open a new issue. Provide a clear description of the enhancement, why it would be beneficial, and any potential implementation ideas you might have.

### Your First Code Contribution

Unsure where to begin contributing? You can start by looking through `good first issue` or `help wanted` issues (if available):

*   **Good first issues** - Issues which should only require a few lines of code, and are good ways to get familiar with the codebase.
*   **Help wanted issues** - Issues which should be somewhat more involved than `good first issue` issues.

### Pull Requests

The process described here has several goals:

*   Maintain code quality.
*   Fix problems that are important to users.
*   Engage the community in working toward the best possible CUDA Fractal Explorer.
*   Enable a sustainable system for maintaining the project.

1.  **Fork the repository** and create your branch from `master` (or the primary development branch).
    *   Use a descriptive branch name, e.g., `feature/add-new-palette` or `fix/nvrtc-compile-error`.
2.  **Set up your development environment** (see [Development Setup](#development-setup)).
3.  **Make your changes.** Ensure your code adheres to the [Style Guide](#style-guide).
4.  **Test your changes thoroughly.**
    *   Run the application and verify your changes work as expected.
    *   Test edge cases (e.g., deep zooms, different palettes, switching between CUDA/NVRTC/CPU modes if applicable).
    *   If possible, test on different hardware (especially different NVIDIA GPUs).
    *   Ensure your changes don't introduce new warnings or errors during compilation.
5.  **Commit your changes.** Use clear and concise commit messages.
6.  **Push your branch** to your fork.
7.  **Open a Pull Request (PR)** to the `master` branch of the original `NeKon69/MandlebrotSetSFML` repository.
8.  **Describe your changes** in the PR description.
    *   Explain the purpose of the changes.
    *   Link to any relevant issues (e.g., `Closes #123`).
    *   Mention any potential drawbacks or areas for further discussion.
9.  **Wait for review.** Respond to any feedback or requests for changes from the maintainers.

## Development Setup

To contribute code, you'll need to be able to build the project from source. Please follow the instructions in the [Getting Started (Building the Project)](https://github.com/NeKon69/MandlebrotSetSFML#getting-started-building-the-project) section of the README.md.

Key requirements include:

*   An NVIDIA GPU (Compute Capability 3.0+)
*   NVIDIA CUDA Toolkit
*   A C++20 compatible compiler (GCC, Clang, MSVC)
*   CMake (3.30+)
*   vcpkg (for SFML and TGUI dependencies)

Remember to configure CMake using the `vcpkg.cmake` toolchain file:

```bash
# Example configuration command (adjust paths)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake
```

**Note:** Using `-DCMAKE_BUILD_TYPE=Debug` is recommended for development as it enables debugging symbols and disables optimizations, making it easier to trace issues. Use `Release` for performance testing and final builds.

Build the project using:

```bash
cmake --build build --config Debug
```

## Style Guide

Consistency is key. Please try to follow the coding style of the existing codebase. Key points include:

*   **Naming Conventions:** Observe the existing use of `snake_case` or `camelCase` for variables, functions, classes, etc., and stick to it.
*   **Indentation:** Use spaces for indentation (commonly 4 spaces). Ensure your editor is configured accordingly.
*   **Comments:** Write comments for complex logic, CUDA kernel intricacies, or non-obvious code sections.
*   **C++20 Features:** Use modern C++ features where appropriate and where they improve clarity or performance.
*   **CUDA Code:** Follow standard CUDA best practices for kernel launches, memory management (use RAII wrappers or smart pointers where applicable), and error checking (`cudaError_t` checking).
*   **Headers:** Include guards should be used. Prefer forward declarations where possible to reduce compile times.
*   **Line Length:** Try to keep lines reasonably short (e.g., under 100-120 characters) for readability.

## Questions?

If you have questions about contributing, feel free to:

*   Open an issue on [GitHub Issues](https://github.com/NeKon69/MandlebrotSetSFML/issues) and tag it as a `question`.
*   Contact the maintainer directly at [nobodqwe@gmail.com](mailto:nobodqwe@gmail.com).

Thank you for contributing!