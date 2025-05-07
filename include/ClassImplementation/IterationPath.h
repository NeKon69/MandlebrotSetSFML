//
// Created by progamers on 5/7/25.
//

#ifndef MANDELBROTPROJECT_ITERATIONPATH_H
#define MANDELBROTPROJECT_ITERATIONPATH_H
#include "FractalClass.cuh"

template <>
void FractalBase<fractals::mandelbrot>::drawIterationLines(sf::Vector2i mouse_pos) {
    int curr_iter = 0;
    double zr = 0.0;
    double zi = 0.0;

    if (max_iterations >= iterationpoints.size()) {
        iterationpoints.resize(max_iterations * 2);
    }

    double cr = mouse_pos.x / zoom_x - x_offset;
    double ci = mouse_pos.y / zoom_y - y_offset;

    while (curr_iter < max_iterations && zr * zr + zi * zi < 4.0) {
        double tmp_zr = zr;
        zr = zr * zr - zi * zi + cr;
        zi = 2.0 * tmp_zr * zi + ci;

        double x = (zr + x_offset) * zoom_x;
        double y = (zi + y_offset) * zoom_y;

        iterationpoints[curr_iter].position = sf::Vector2f(x, y);
        iterationpoints[curr_iter].color = sf::Color::Red;
        curr_iter++;
    }

    iterationline.create(curr_iter);
    iterationline.update(iterationpoints.data());
    drawen_iterations = curr_iter;
}

template <>
void FractalBase<fractals::julia>::drawIterationLines(sf::Vector2i mouse_pos) {
    int curr_iter = 0;
    double zr = x_offset + (mouse_pos.x / zoom_x);
    double zi = y_offset + (mouse_pos.y / zoom_y);

    // Calculate cr and ci based on the mouse position and current view parameters
    double cr = zreal;
    double ci = zimag;

    while (curr_iter < max_iterations && zr * zr + zi * zi < 4.0) {
        double tmp_zr = zr;
        zr = zr * zr - zi * zi + cr;
        zi = 2.0 * tmp_zr * zi + ci;

        // Map the complex number z back to screen coordinates
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
