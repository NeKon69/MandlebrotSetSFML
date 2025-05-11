//
// Created by progamers on 5/7/25.
//

#ifndef MANDELBROTPROJECT_ITERATIONPATH_H
#define MANDELBROTPROJECT_ITERATIONPATH_H
#include "HardCodedVars.h"
#include "FractalClass.cuh"

/// Specialization for Mandelbrot: Calculates and updates the iteration path
/// for the point in the complex plane corresponding to the mouse position.
/// This path shows the sequence of z values in the Mandelbrot iteration z = z^2 + c.
template <>
void FractalBase<fractals::mandelbrot>::drawIterationLines(sf::Vector2i mouse_pos) {
    int curr_iter = 0;
    double zr = 0.0; // z0_real
    double zi = 0.0; // z0_imag

    /// Ensure the iterationpoints vector is large enough.
    if (max_iterations >= iterationpoints.size()) {
        iterationpoints.resize(max_iterations * 2);
    }

    /// Convert mouse position to complex coordinate 'c'.
    double cr = mouse_pos.x / zoom_x - x_offset;
    double ci = mouse_pos.y / zoom_y - y_offset;

    while (curr_iter < max_iterations && zr * zr + zi * zi < 4.0) {
        double tmp_zr = zr;
        zr = zr * zr - zi * zi + cr;
        zi = 2.0 * tmp_zr * zi + ci;

        /// Convert the current iterated complex value (zr, zi) back to screen coordinates.
        double x = (zr + x_offset) * zoom_x;
        double y = (zi + y_offset) * zoom_y;

        iterationpoints[curr_iter].position = sf::Vector2f(x, y);
        iterationpoints[curr_iter].color = sf::Color::Red;
        curr_iter++;
    }

    /// Update the SFML vertex buffer with the calculated path.
    iterationline.create(curr_iter);
    iterationline.update(iterationpoints.data());
    drawen_iterations = curr_iter;
}

/// Specialization for Julia: Calculates and updates the iteration path
/// for the point in the complex plane corresponding to the mouse position.
/// This path shows the sequence of z values in the Julia iteration z = z^2 + c,
/// where z0 is determined by the mouse and c is fixed.
template <>
void FractalBase<fractals::julia>::drawIterationLines(sf::Vector2i mouse_pos) {
    int curr_iter = 0;

    /// Initial z0 is determined by the mouse position in the complex plane.
    double zr = mouse_pos.x / zoom_x - x_offset;
    double zi = mouse_pos.y / zoom_y - y_offset;

    // Get current Julia parameters
    double cr = zreal;
    double ci = zimag;

    while (curr_iter < max_iterations && zr * zr + zi * zi < 4.0) {
        double tmp_zr = zr;
        zr = zr * zr - zi * zi + cr;
        zi = 2.0 * tmp_zr * zi + ci;

        /// Map the current iterated complex value (zr, zi) back to screen coordinates.
        float x = static_cast<float>((zr - x_offset) * zoom_x);
        float y = static_cast<float>((zi - y_offset) * zoom_y);

        if (curr_iter < iterationpoints.size()) {
            iterationpoints[curr_iter].position = sf::Vector2f(x, y);
            iterationpoints[curr_iter].color = sf::Color::Red;
        }

        curr_iter++;
    }

    iterationline.create(curr_iter);
    iterationline.update(iterationpoints.data());
    drawen_iterations = curr_iter;
}

#endif //MANDELBROTPROJECT_ITERATIONPATH_H