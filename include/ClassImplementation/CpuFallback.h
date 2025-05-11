//
// Created by progamers on 5/6/25.
//
#pragma once
#include "HardCodedVars.h"
#include "FractalClass.cuh"
#include <iostream>
template <typename Derived>
    void FractalBase<Derived>::set_grid(dim3 block) {
    dimBlock = block;
    dimGrid = dim3(
            (width + dimBlock.x - 1) / dimBlock.x,
            (height + dimBlock.y - 1) / dimBlock.y
    );
}

void cpu_render_mandelbrot(render_target target, unsigned char* pixels, unsigned int width, unsigned int height, double zoom_x, double zoom_y,
                           double x_offset, double y_offset, Color* palette, unsigned int paletteSize,
                           unsigned int max_iterations, unsigned int* total_iterations, std::atomic<unsigned char>& finish_flag
)
{
    try {
        unsigned int total_iterations_local = 0;
        finish_flag.store(0);
        if (target.x_end > width) {
            target.x_end = width;
        }
        if (target.y_end > height) {
            target.y_end = height;
        }
        if (target.x_start < 0) {
            target.x_start = 0;
        }
        if (target.y_start < 0) {
            target.y_start = 0;
        }
        for(unsigned int y = target.y_start; y < target.y_end; ++y){
            for(unsigned int x = target.x_start; x < target.x_end; ++x){
                double zr = 0.0;
                double zi = 0.0;
                double cr = x / zoom_x - x_offset;
                double ci = y / zoom_y - y_offset;
                unsigned char r, g, b;
                float curr_iter = 0;
                while (curr_iter < max_iterations && zr * zr + zi * zi < 100.0) {
                    double tmp_zr = zr;
                    zr = zr * zr - zi * zi + cr;
                    zi = 2.0 * tmp_zr * zi + ci;

                    ++curr_iter;
                    if(finish_flag.load() == 2) {
                        finish_flag.store(1);
                        return;
                    }
                }
                if (curr_iter == max_iterations) {
                    r = g = b = 0;
                } else {
                    double smooth_iteration = curr_iter + 1.0f - log2f(log2f(sqrtf(zr * zr + zi * zi)));

                    const double cycle_scale_factor = 25.0f;
                    double virtual_pos = smooth_iteration * cycle_scale_factor;

                    double normalized_pingpong = fmodf(virtual_pos / static_cast<double>(paletteSize -1), 2.0f);
                    if (normalized_pingpong < 0.0f) {
                        normalized_pingpong += 2.0f;
                    }

                    double t_interp;
                    if (normalized_pingpong <= 1.0f) {
                        t_interp = normalized_pingpong;
                    } else {
                        t_interp = 2.0f - normalized_pingpong;
                    }

                    double float_index = t_interp * (paletteSize - 1);

                    int index1 = static_cast<int>(floorf(float_index));
                    int index2 = std::min(int(paletteSize - 1), index1 + 1);

                    index1 = std::max(0, index1);

                    double t_local = fmodf(float_index, 1.0f);
                    if (t_local < 0.0f) t_local += 1.0f;

                    Color color1 = getPaletteColor(index1, paletteSize, palette);
                    Color color2 = getPaletteColor(index2, paletteSize, palette);

                    float r_f = static_cast<float>(color1.r) + t_local * (static_cast<float>(color2.r) - static_cast<float>(color1.r));
                    float g_f = static_cast<float>(color1.g) + t_local * (static_cast<float>(color2.g) - static_cast<float>(color1.g));
                    float b_f = static_cast<float>(color1.b) + t_local * (static_cast<float>(color2.b) - static_cast<float>(color1.b));

                    r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r_f)));
                    g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g_f)));
                    b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b_f)));
                }
                const unsigned int index = (y * width + x) * 4;
                pixels[index] = r;
                pixels[index + 1] = g;
                pixels[index + 2] = b;
                pixels[index + 3] = 255;
                total_iterations_local += curr_iter;
            }
        }
        *total_iterations += total_iterations_local;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR in cpu_render_mandelbrot thread (target y=" << target.y_start << "): " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "ERROR in cpu_render_mandelbrot thread (target y=" << target.y_start << "): Unknown exception!" << std::endl;
    }
    finish_flag.store(1);
}

void cpu_render_julia(render_target target, unsigned char* pixels, unsigned int width, unsigned int height, double zoom_x, double zoom_y,
                      double x_offset, double y_offset, Color* palette, unsigned int paletteSize,
                      unsigned int max_iterations, unsigned int* total_iterations, std::atomic<unsigned char>& finish_flag,
                      double seed_real, double seed_imag
)
{
    try {
        unsigned int total_iterations_local = 0;
        finish_flag.store(0);
        if (target.x_end > width) {
            target.x_end = width;
        }
        if (target.y_end > height) {
            target.y_end = height;
        }
        if (target.x_start < 0) {
            target.x_start = 0;
        }
        if (target.y_start < 0) {
            target.y_start = 0;
        }
        for(unsigned int y = target.y_start; y < target.y_end; ++y){
            for(unsigned int x = target.x_start; x < target.x_end; ++x){
                double zr = x / zoom_x - x_offset;
                double zi = y / zoom_y - y_offset;
                unsigned char r, g, b;
                float curr_iter = 0;
                while (curr_iter < max_iterations && zr * zr + zi * zi < 100.0) {
                    double tmp_zr = zr;
                    zr = zr * zr - zi * zi + seed_real;
                    zi = 2.0 * tmp_zr * zi + seed_imag;

                    ++curr_iter;
                    if(finish_flag.load() == 2) {
                        finish_flag.store(1);
                        return;
                    }
                }
                if (curr_iter == max_iterations) {
                    r = g = b = 0;
                } else {
                    double smooth_iteration = curr_iter + 1.0f - log2f(log2f(sqrtf(zr * zr + zi * zi)));

                    const double cycle_scale_factor = 25.0f;
                    double virtual_pos = smooth_iteration * cycle_scale_factor;

                    double normalized_pingpong = fmodf(virtual_pos / static_cast<double>(paletteSize -1), 2.0f);
                    if (normalized_pingpong < 0.0f) {
                        normalized_pingpong += 2.0f;
                    }

                    double t_interp;
                    if (normalized_pingpong <= 1.0f) {
                        t_interp = normalized_pingpong;
                    } else {
                        t_interp = 2.0f - normalized_pingpong;
                    }

                    double float_index = t_interp * (paletteSize - 1);

                    int index1 = static_cast<int>(floorf(float_index));
                    int index2 = std::min(int(paletteSize - 1), index1 + 1);

                    index1 = std::max(0, index1);

                    double t_local = fmodf(float_index, 1.0f);
                    if (t_local < 0.0f) t_local += 1.0f;

                    Color color1 = getPaletteColor(index1, paletteSize, palette);
                    Color color2 = getPaletteColor(index2, paletteSize, palette);

                    float r_f = static_cast<float>(color1.r) + t_local * (static_cast<float>(color2.r) - static_cast<float>(color1.r));
                    float g_f = static_cast<float>(color1.g) + t_local * (static_cast<float>(color2.g) - static_cast<float>(color1.g));
                    float b_f = static_cast<float>(color1.b) + t_local * (static_cast<float>(color2.b) - static_cast<float>(color1.b));

                    r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r_f)));
                    g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g_f)));
                    b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b_f)));
                }
                const unsigned int index = (y * width + x) * 4;
                pixels[index] = r;
                pixels[index + 1] = g;
                pixels[index + 2] = b;
                pixels[index + 3] = 255;
                total_iterations_local += curr_iter;
            }
        }
        *total_iterations += total_iterations_local;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR in cpu_render_julia thread (target y=" << target.y_start << "): " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "ERROR in cpu_render_julia thread (target y=" << target.y_start << "): Unknown exception!" << std::endl;
    }
    finish_flag.store(1);
}
