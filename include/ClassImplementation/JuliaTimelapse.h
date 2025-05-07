//
// Created by progamers on 5/6/25.
//
#pragma once
#include "ClassImplementation/FractalClass.cuh"
template <>
void FractalBase<fractals::julia>::start_timelapse() {
    timelapse.zx = disX(gen);
    timelapse.zy = disY(gen);
    timelapse.velocityX = disVelX(gen);
    timelapse.velocityY = disVelY(gen);
}

template <>
void FractalBase<fractals::julia>::update_timelapse() {
    const auto elapsed = clock.getElapsedTime();
    const auto elapsedMs = elapsed.asMilliseconds();

    constexpr float frameTimeMs = 1000.0f / 360.0f;

    if (elapsedMs <= frameTimeMs) return;

    float deltaTime = elapsedMs / 1000.0f;
    clock.restart();

    constexpr float GRAVITATIONAL_STRENGTH = 0.1f;
    constexpr float EVENT_HORIZON = 0.1f;
    constexpr float MAX_VEL = 20.0f;
    constexpr float VELOCITY_DAMPING = 0.9995f;
    constexpr float TARGET_CHANGE_TIME = 2.0f;
    constexpr float TRANSITION_SPEED = 3.5f;

    static float timeToTargetChange = TARGET_CHANGE_TIME;
    static float targetZX = 0.0f;
    static float targetZY = 0.0f;
    static float currentTargetZX = 0.0f;
    static float currentTargetZY = 0.0f;

    timeToTargetChange -= deltaTime;
    if (timeToTargetChange <= 0) {
        timeToTargetChange = TARGET_CHANGE_TIME;
        targetZX = (static_cast<float>(rand()) / RAND_MAX) * 1.0f - 0.5f;
        targetZY = (static_cast<float>(rand()) / RAND_MAX) * 1.0f - 0.5f;
    }

    currentTargetZX += (targetZX - currentTargetZX) * TRANSITION_SPEED * deltaTime;
    currentTargetZY += (targetZY - currentTargetZY) * TRANSITION_SPEED * deltaTime;

    float dx = currentTargetZX - timelapse.zx;
    float dy = currentTargetZY - timelapse.zy;
    float distanceSq = dx * dx + dy * dy + 0.0001f;

    timelapse.velocityX += GRAVITATIONAL_STRENGTH * dx * deltaTime;
    timelapse.velocityY += GRAVITATIONAL_STRENGTH * dy * deltaTime;

    float distance = sqrt(distanceSq);
    if (distance < EVENT_HORIZON) {
        float angle = atan2(dy, dx);
        float boost = (1.0f - (distance / EVENT_HORIZON)) * 4.0f;
        timelapse.velocityX += cos(angle + 3.14159265359f / 2.0f) * boost * deltaTime;
        timelapse.velocityY += sin(angle + 3.14159265359f / 2.0f) * boost * deltaTime;
    }

    timelapse.velocityX *= VELOCITY_DAMPING;
    timelapse.velocityY *= VELOCITY_DAMPING;
    timelapse.velocityX = std::clamp(timelapse.velocityX, -MAX_VEL, MAX_VEL);
    timelapse.velocityY = std::clamp(timelapse.velocityY, -MAX_VEL, MAX_VEL);

    timelapse.zx += timelapse.velocityX * deltaTime;
    timelapse.zy += timelapse.velocityY * deltaTime;

    constexpr float BOUNDARY = 2.0f;
    if (std::abs(timelapse.zx) > BOUNDARY) {
        timelapse.zx = std::copysign(BOUNDARY, timelapse.zx);
        timelapse.velocityX *= -0.5f;
    }
    if (std::abs(timelapse.zy) > BOUNDARY) {
        timelapse.zy = std::copysign(BOUNDARY, timelapse.zy);
        timelapse.velocityY *= -0.5f;
    }

    render(render_state::good, timelapse.zx, timelapse.zy);
}

template <>
void FractalBase<fractals::julia>::stop_timelapse() {
    timelapse.zx = 0;
    timelapse.zy = 0;
    timelapse.velocityX = 0;
    timelapse.velocityY = 0;
}