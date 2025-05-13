//
// Created by progamers on 5/6/25.
//
#pragma once
#include "HardCodedVars.h"
#include "ClassImplementation/FractalClass.cuh"
#include "ClassImplementation/Macros.h"
#include "ClassImplementation/CpuFallback.h"

/// Renders the Julia set with specified parameters `zx` and `zy` (for the constant c).
/// This function handles both GPU (CUDA Runtime and NVRTC) and CPU fallback rendering paths.
template <>
void FractalBase<fractals::julia>::render(
        render_state quality,
        double zx, double zy
) {
    zreal = zx; // Store Julia constant c_real
    zimag = zy; // Store Julia constant c_imag

    if (!isCudaAvailable) {
        // --- CPU Rendering Fallback ---
        if (quality == render_state::best) { // CPU 'best' quality is not implemented.
            return;
        }


        /// Global lock to ensure only one CPU render is active.
        bool expected_state = false;
        if (!g_isCpuRendering.compare_exchange_strong(expected_state, true, std::memory_order_acq_rel)) {
            return;
        }
        *h_total_iterations = 0; // Reset total iterations for CPU rendering.
        /// Launch CPU rendering in a detached background thread.
        std::thread main_thread([&]() {
            RenderGuard renderGuard(g_isCpuRendering); // RAII lock release.

            unsigned int max_threads_local = std::thread::hardware_concurrency();
            if (max_threads_local == 0) max_threads_local = 1;
            render_targets.clear();

            /// Divide rendering task for CPU threads.
            unsigned int rows_per_thread = basic_height / max_threads_local;
            unsigned int remainder_rows = basic_height % max_threads_local;
            unsigned int current_y_start = 0;
            for (unsigned int i = 0; i < max_threads_local; ++i) {
                unsigned int rows_for_this_thread = rows_per_thread + (i < remainder_rows ? 1 : 0);
                if (rows_for_this_thread == 0 && current_y_start >= basic_height) continue;
                unsigned int y_start = current_y_start;
                unsigned int y_end = current_y_start + rows_for_this_thread;
                if (y_start < y_end) {
                    render_targets.emplace_back(0, y_start, basic_width, y_end);
                    current_y_start = y_end;
                } else if (current_y_start >= basic_height) {
                    break;
                }
            }
            unsigned int actual_threads_to_launch = render_targets.size();

            if (actual_threads_to_launch > thread_stop_flags.size()) {
                std::cerr << "Warning: Clamping required threads (" << actual_threads_to_launch
                          << ") to available flags (" << thread_stop_flags.size() << ")." << std::endl;
                actual_threads_to_launch = thread_stop_flags.size();
            }
            if (actual_threads_to_launch == 0 && basic_height > 0 && basic_width > 0) {
                return;
            }

            std::mutex mutex;
            /// Launch CPU worker threads.
            for (unsigned int i = 0; i < actual_threads_to_launch; ++i) {
                thread_stop_flags[i].store(0, std::memory_order_release);
                std::thread t(cpu_render_julia, render_targets[i], pixels, basic_width, basic_height,
                              zoom_x, zoom_y, x_offset, y_offset, palette.data(), paletteSize,
                              max_iterations, h_total_iterations, std::ref(thread_stop_flags[i]), zreal, zimag, std::ref(mutex));
                t.detach();
            }

            /// Poll worker thread completion and call post_processing.
            while (true) {
                bool all_done = true;
                if (actual_threads_to_launch > 0) {
                    for(unsigned int i = 0; i < actual_threads_to_launch; ++i) {
                        if (thread_stop_flags[i].load(std::memory_order_acquire) != 1) {
                            all_done = false;
                            break;
                        }
                    }
                }
                if (all_done) break;
                post_processing();
                std::this_thread::sleep_for(THREAD_SLEEP_TIME);
            }
            post_processing();
            std::cout << "Finished thread (Julia)!\n";
        });
        main_thread.detach();
        return;
    } // End of CPU Fallback

    /// GPU Rendering Path
    RECALCULATE_RENDER_PARAMS(quality);

    if(!custom_formula){ // Standard CUDA Runtime API path
        if (render_zoom_x > DOUBLE_PRECISION_ZOOM_THRESHOLD) { // Double precision
            set_grid(BLOCK_SIZE_DOUBLE);
            fractal_rendering<double><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations, zx, zy
            );
        }
        else { // Single precision
            set_grid(BLOCK_SIZE_FLOAT);
            fractal_rendering<float><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations, zx, zy
            );
        }
    }
    else { // NVRTC / Custom Formula path (CUDA Driver API)
        CU_SAFE_CALL(cuCtxSetCurrent(ctx));

        double render_zoom_x_d = render_zoom_x;
        double render_zoom_y_d = render_zoom_y;
        double x_offset_d = x_offset;
        double y_offset_d = y_offset;
        float max_iterations_val_f = static_cast<float>(max_iterations);
        double max_iterations_val_d = static_cast<double>(max_iterations);
        unsigned int paletteSize_val = paletteSize;
        unsigned int width_val = width;
        unsigned int height_val = height;
        /// Pass Julia constants `zx` (c_real) and `zy` (c_imag) to the kernel.
        double zx_d = zx;
        double zy_d = zy;


        if (zoom_x > DOUBLE_PRECISION_ZOOM_THRESHOLD) { // NVRTC Double precision
            set_grid(BLOCK_SIZE_DOUBLE);
            void* args[] = {
                    &cu_d_pixels, &len, &width_val, &height_val,
                    &render_zoom_x_d, &render_zoom_y_d, &x_offset_d, &y_offset_d,
                    &cu_palette, &paletteSize_val, &max_iterations_val_d, &cu_d_total_iterations,
                    &zx_d, &zy_d // Julia constants
            };
            CU_SAFE_CALL(cuLaunchKernel(kernelDouble, dimGrid.x, dimGrid.y, 1,
                                        dimBlock.x, dimBlock.y, 1,
                                        0, CUss, args, nullptr));
        } else { // NVRTC Single precision
            set_grid(BLOCK_SIZE_FLOAT);
            float render_zoom_x_f = static_cast<float>(render_zoom_x_d);
            float render_zoom_y_f = static_cast<float>(render_zoom_y_d);
            float x_offset_f = static_cast<float>(x_offset_d);
            float y_offset_f = static_cast<float>(y_offset_d);
            float zx_f = static_cast<float>(zx_d);
            float zy_f = static_cast<float>(zy_d);

            void* args[] = {
                    &cu_d_pixels, &len, &width_val, &height_val,
                    &render_zoom_x_f, &render_zoom_y_f, &x_offset_f, &y_offset_f,
                    &cu_palette, &paletteSize_val, &max_iterations_val_f, &cu_d_total_iterations,
                    &zx_f, &zy_f // Julia constants
            };
            CU_SAFE_CALL(cuLaunchKernel(kernelFloat, dimGrid.x, dimGrid.y, 1,
                                        dimBlock.x, dimBlock.y, 1,
                                        0, CUss, args, nullptr));
        }
    }

    /// Error handling and retry logic for CUDA Runtime kernels.
    cudaError_t err = cudaGetLastError();
    while (err != cudaSuccess && context != context_type::NVRTC) { // Only retry for CUDA Runtime
        set_grid({dimBlock.x - BLOCK_SUBSTRACTION, dimBlock.y - BLOCK_SUBSTRACTION});
        if (zoom_x > DOUBLE_PRECISION_ZOOM_THRESHOLD) {
            fractal_rendering<double><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations, zx, zy
            );
        }
        else {
            fractal_rendering<float><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations, zx, zy
            );
        }
        err = cudaGetLastError();
        if (dimBlock.x < BLOCK_SUBSTRACTION + 1) {
            std::cerr << "Critical Issue (julia set): " << cudaGetErrorString(err) << "\n";
            return;
        }
    }
    post_processing();
}