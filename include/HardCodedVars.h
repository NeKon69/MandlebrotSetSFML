//
// Created by progamers on 5/11/25.
//
#pragma once

#include <string>
#include <chrono>

// This is how long the CPU main thread will sleep until updating the UI in microseconds
inline constexpr std::chrono::microseconds THREAD_SLEEP_TIME(1);

// Amount of zoom needed to switch to the double precision mode
// It varies also on how hard operations you are performing
// for example addition isn't that hard but sin is
inline constexpr float DOUBLE_PRECISION_ZOOM_THRESHOLD = 1e7f;

// Block size for float
inline constexpr dim3 BLOCK_SIZE_FLOAT(32, 32, 1);

// Block size for double
inline constexpr dim3 BLOCK_SIZE_DOUBLE(10, 10, 1);

// Block size for antialiasing can be the same as float one cause it performs even easier operations then mandelbrot kernel but we will still define it for fun
inline constexpr dim3 BLOCK_SIZE_ANTIALIASING(32, 32, 1);

// Amount of block we subtract from the current blocks
inline constexpr unsigned int BLOCK_SUBSTRACTION = 2;

// Minimal zoom factor, basically can be infinite, but we set it to 1e-2 for not zooming out too much
inline constexpr double MIN_ZOOM_FACTOR = 0.01;

// Maximal zoom factor, basically can be infinite, but we set it to 1e14 for double precision limitations
inline constexpr double MAX_ZOOM_FACTOR = 100000000000000.0;

// Timelapse properties
inline constexpr float GRAVITATIONAL_STRENGTH = 0.1f;
inline constexpr float EVENT_HORIZON = 0.1f;
inline constexpr float MAX_VEL = 20.0f;
inline constexpr float VELOCITY_DAMPING = 0.9995f;
inline constexpr float TARGET_CHANGE_TIME = 2.0f;
inline constexpr float TRANSITION_SPEED = 3.5f;
inline constexpr float frameTimeMs = 1000.0f / 360.0f;

// Fractal and image properties
inline constexpr unsigned int MAX_ITERATIONS = 300;
inline constexpr unsigned int BASIC_WIDTH = 800;
inline constexpr unsigned int BASIC_HEIGHT = 600;
inline constexpr double BASIC_X_OFFSET = 2.25;
inline constexpr double BASIC_Y_OFFSET = 1.25;
inline constexpr double BASIC_ZOOM_X = 240.0;
inline constexpr double BASIC_ZOOM_Y = 240.0;
inline constexpr double BASIC_ZOOM_SPEED = 0.1;
inline constexpr double BASIC_ZOOM_SCALE = 1.0;
inline constexpr double BASIC_ZOOM_FACTOR = 1.0;
inline constexpr double BASIC_MAX_COMPUTATION_F = 50.f;
inline constexpr double BASIC_MAX_COMPUTATION_D = 50.f;
inline constexpr unsigned int BASIC_PALETTE_SIZE = 20000;