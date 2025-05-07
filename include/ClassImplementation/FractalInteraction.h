//
// Created by progamers on 5/6/25.
//
#pragma once
#include "ClassImplementation/FractalClass.cuh"
// Mouse pos should be relative to the picture and not to the screen
template <typename Derived>
void FractalBase<Derived>::handleZoom(double wheel_delta, const sf::Vector2i mouse_pos) {

    double old_zoom_x = zoom_x;
    double old_zoom_y = zoom_y;
    double old_x_offset = x_offset;
    double old_y_offset = y_offset;

    double zoom_change = 1.0 + wheel_delta * zoom_speed;
    zoom_factor *= zoom_change;
    zoom_factor = std::max(std::min(zoom_factor, 100000000000000.0), 0.01);

    zoom_x = basic_zoom_x * zoom_factor;
    zoom_y = basic_zoom_y * zoom_factor;

    double image_mouse_x = mouse_pos.x * 1.0;
    double image_mouse_y = mouse_pos.y * 1.0;


    x_offset = old_x_offset + (image_mouse_x / zoom_x - image_mouse_x / old_zoom_x);
    y_offset = old_y_offset + (image_mouse_y / zoom_y - image_mouse_y / old_zoom_y);

}

template <typename Derived>
void FractalBase<Derived>::start_dragging(sf::Vector2i mouse_pos) {
    is_dragging = true;
    drag_start_pos = mouse_pos;
}


template <typename Derived>
void FractalBase<Derived>::dragging(sf::Vector2i mouse_pos) {
    if (!is_dragging) return;

    sf::Vector2i delta_pos = mouse_pos - drag_start_pos;
    double delta_real = static_cast<double>(delta_pos.x) / (zoom_x * zoom_scale);
    double delta_imag = static_cast<double>(delta_pos.y) / (zoom_y * zoom_scale);

    x_offset += delta_real;
    y_offset += delta_imag;
    drag_start_pos = mouse_pos;
}

template <typename Derived>
void FractalBase<Derived>::stop_dragging() {
    is_dragging = false;
}

template <typename Derived>
void FractalBase<Derived>::move_fractal(sf::Vector2i offset) {
    x_offset += offset.x / (zoom_x * zoom_scale);
    y_offset += offset.y / (zoom_y * zoom_scale);
}