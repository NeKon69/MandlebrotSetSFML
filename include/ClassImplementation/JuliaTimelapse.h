//
// Created by progamers on 5/6/25.
//
#pragma once
#include "HardCodedVars.h"
#include "FractalClass.cuh"

/// Specialization of start_timelapse for the Julia fractal.
/// Initializes the Julia set parameters (cReal, cImaginary) and their
/// velocities with random values to begin the animation.
template <>
void FractalBase<fractals::julia>::start_timelapse() {
    timelapse.zx = disX(gen);
    timelapse.zy = disY(gen);
    timelapse.velocityX = disVelX(gen);
    timelapse.velocityY = disVelY(gen);
}

/// Specialization of update_timelapse for the Julia fractal.
/// Advances the Julia set parameters (cReal, cImaginary) based on a
/// simulated physical system, creating a dynamic animation.
template <>
void FractalBase<fractals::julia>::update_timelapse() {
    const auto elapsed = clock.getElapsedTime();
    const auto elapsedMs = elapsed.asMilliseconds();

    /// Control the frame rate of the timelapse updates.
    if (elapsedMs <= frameTimeMs) return;

    float deltaTime = elapsedMs / 1000.0f;
    clock.restart();

    /// Static variables to manage a target point in the complex plane
    /// that the Julia parameters are attracted to. The target changes
    /// periodically.
    static float timeToTargetChange = TARGET_CHANGE_TIME;
    static float targetZX = 0.0f;
    static float targetZY = 0.0f;
    static float currentTargetZX = 0.0f;
    static float currentTargetZY = 0.0f;

    /// Update the target point periodically.
    timeToTargetChange -= deltaTime;
    if (timeToTargetChange <= 0) {
        timeToTargetChange = TARGET_CHANGE_TIME;
        targetZX = (static_cast<float>(rand()) / RAND_MAX) * 1.0f - 0.5f;
        targetZY = (static_cast<float>(rand()) / RAND_MAX) * 1.0f - 0.5f;
    }

    /// Smoothly transition the current target towards the new random target.
    currentTargetZX += (targetZX - currentTargetZX) * TRANSITION_SPEED * deltaTime;
    currentTargetZY += (targetZY - currentTargetZY) * TRANSITION_SPEED * deltaTime;

    /// Calculate the vector from the current Julia parameters to the smoothly transitioning target.
    float dx = currentTargetZX - timelapse.zx;
    float dy = currentTargetZY - timelapse.zy;
    float distanceSq = dx * dx + dy * dy + 0.0001f; // Add small epsilon to avoid division by zero

    /// Apply a "gravitational" force pulling the Julia parameters towards the target.
    timelapse.velocityX += GRAVITATIONAL_STRENGTH * dx * deltaTime;
    timelapse.velocityY += GRAVITATIONAL_STRENGTH * dy * deltaTime;

    /// Implement an "event horizon" effect: if the parameters get too close
    /// to the target, apply a perpendicular boost to create more dynamic movement
    /// and prevent settling exactly on the target.
    float distance = sqrt(distanceSq);
    if (distance < EVENT_HORIZON) {
        float angle = atan2(dy, dx);
        float boost = (1.0f - (distance / EVENT_HORIZON)) * 4.0f;
        timelapse.velocityX += cos(angle + 3.14159265359f / 2.0f) * boost * deltaTime; // Perpendicular direction
        timelapse.velocityY += sin(angle + 3.14159265359f / 2.0f) * boost * deltaTime; // Perpendicular direction
    }

    /// Apply damping to velocities and clamp them to a maximum value.
    timelapse.velocityX *= VELOCITY_DAMPING;
    timelapse.velocityY *= VELOCITY_DAMPING;
    timelapse.velocityX = std::clamp(timelapse.velocityX, -MAX_VEL, MAX_VEL);
    timelapse.velocityY = std::clamp(timelapse.velocityY, -MAX_VEL, MAX_VEL);

    /// Update the Julia parameters based on velocities and delta time.
    timelapse.zx += timelapse.velocityX * deltaTime;
    timelapse.zy += timelapse.velocityY * deltaTime;

    /// Implement boundary checks: if parameters exceed a certain range,
    /// clamp them and reverse/damp their velocity to keep them within bounds.
    constexpr float BOUNDARY = 2.0f;
    if (std::abs(timelapse.zx) > BOUNDARY) {
        timelapse.zx = std::copysign(BOUNDARY, timelapse.zx);
        timelapse.velocityX *= -0.5f;
    }
    if (std::abs(timelapse.zy) > BOUNDARY) {
        timelapse.zy = std::copysign(BOUNDARY, timelapse.zy);
        timelapse.velocityY *= -0.5f;
    }

    /// Render the Julia set with the updated parameters using 'good' quality.
    render(render_state::good, timelapse.zx, timelapse.zy);
}

/// Specialization of stop_timelapse for the Julia fractal.
/// Resets the Julia set parameters and velocities to zero, stopping the animation.
template <>
void FractalBase<fractals::julia>::stop_timelapse() {
    timelapse.zx = 0;
    timelapse.zy = 0;
    timelapse.velocityX = 0;
    timelapse.velocityY = 0;
}