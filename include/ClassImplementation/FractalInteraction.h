//
// Created by progamers on 5/6/25.
//
#pragma once
#include "HardCodedVars.h"
#include "FractalClass.cuh"

/// Handles zoom functionality for the fractal view.
/// Mouse position (`mouse_pos`) is relative to the fractal image, not the screen.
/// Adjusts the zoom factor and recalculates the view offset to keep the point
/// under the mouse cursor stationary.
template <typename Derived>
void FractalBase<Derived>::handleZoom(double wheel_delta, const sf::Vector2i mouse_pos) {

    double old_zoom_x = zoom_x;
    double old_zoom_y = zoom_y;
    double old_x_offset = x_offset;
    double old_y_offset = y_offset;

    double zoom_change = 1.0 + wheel_delta * zoom_speed;
    zoom_factor *= zoom_change;
    /// Clamp the zoom factor to prevent extreme zoom levels.
    zoom_factor = std::max(std::min(zoom_factor, MAX_ZOOM_FACTOR), MIN_ZOOM_FACTOR);

    zoom_x = basic_zoom_x * zoom_factor;
    zoom_y = basic_zoom_y * zoom_factor;

    double image_mouse_x = mouse_pos.x;
    double image_mouse_y = mouse_pos.y;

    /// Recalculate offsets to keep the point under the mouse cursor fixed during zoom.
    /// This involves finding the complex coordinate under the mouse before and after
    /// the zoom and adjusting the offset to compensate for the change.
    x_offset = old_x_offset + (image_mouse_x / zoom_x - image_mouse_x / old_zoom_x);
    y_offset = old_y_offset + (image_mouse_y / zoom_y - image_mouse_y / old_zoom_y);
}

/// Initiates a dragging operation.
/// Stores the initial mouse position to calculate subsequent drag deltas.
template <typename Derived>
void FractalBase<Derived>::start_dragging(sf::Vector2i mouse_pos) {
    is_dragging = true;
    drag_start_pos = mouse_pos;
}

/// Handles the dragging operation while the mouse is moved.
/// Calculates the change in mouse position and updates the view offset accordingly.
template <typename Derived>
void FractalBase<Derived>::dragging(sf::Vector2i mouse_pos) {
    if (!is_dragging) return;

    sf::Vector2i delta_pos = mouse_pos - drag_start_pos;
    /// Convert pixel delta to complex plane delta based on current zoom.
    double delta_real = static_cast<double>(delta_pos.x) / (zoom_x * zoom_scale);
    double delta_imag = static_cast<double>(delta_pos.y) / (zoom_y * zoom_scale);

    x_offset += delta_real;
    y_offset += delta_imag;
    /// Update the drag start position for the next delta calculation.
    drag_start_pos = mouse_pos;
}

/// Stops the dragging operation.
template <typename Derived>
void FractalBase<Derived>::stop_dragging() {
    is_dragging = false;
}

/// Moves the fractal view by a specified pixel offset.
/// Used for keyboard-based navigation.
template <typename Derived>
void FractalBase<Derived>::move_fractal(sf::Vector2i offset) {
    x_offset += offset.x / (zoom_x * zoom_scale);
    y_offset += offset.y / (zoom_y * zoom_scale);
}