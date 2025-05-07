//
// Created by progamers on 5/6/25.
//
#pragma once
#include "ClassImplementation/FractalClass.cuh"
#include "ClassImplementation/Macros.h"
#include <iostream>
#include "ClassImplementation/CpuFallback.h"
template <>
void FractalBase<fractals::mandelbrot>::render(render_state quality) {
    if (!isCudaAvailable) {
        // --- CPU Rendering Fallback ---
        // This block executes only if CUDA (GPU acceleration) is not available.

        if (quality == render_state::best) {
            return;
        }

        // --- Global CPU Rendering Lock ---
        // Use an atomic flag to prevent multiple CPU rendering processes from running concurrently.
        bool expected_state = false; // We expect no CPU render to be currently running.
        // Attempt to atomically set g_isCpuRendering to true IF it's currently false.
        // memory_order_acq_rel: Ensures memory operations before this in other threads are visible (acquire)
        //                      and memory operations after this in this thread are visible to others (release).
        //                      Crucial for coordinating access to shared rendering resources.
        if (!g_isCpuRendering.compare_exchange_strong(expected_state, true, std::memory_order_acq_rel)) {
            // If the exchange failed, it means g_isCpuRendering was already true. Another CPU render is active.
            return;
        }

        // --- Launch Background CPU Rendering Thread ---
        // Offload the CPU rendering to a separate thread to avoid blocking the main application thread.
        std::thread main_thread([&]() { // Lambda captures necessary variables by reference.

            // --- Automatic Lock Release ---
            // This RAII guard ensures g_isCpuRendering is set back to false when the lambda finishes
            // execution, even if an exception occurs. This releases the global rendering lock.
            RenderGuard renderGuard(g_isCpuRendering);

            // --- Prepare for New Render Task & Divide Work ---
            unsigned int max_threads_local = std::thread::hardware_concurrency();
            if (max_threads_local == 0) max_threads_local = 1;
            render_targets.clear(); // Clear previous task definitions.

            // Distribute image rows among available threads, handling remainders.
            unsigned int rows_per_thread = basic_height / max_threads_local;
            unsigned int remainder_rows = basic_height % max_threads_local;
            unsigned int current_y_start = 0;
            for (unsigned int i = 0; i < max_threads_local; ++i) {
                unsigned int rows_for_this_thread = rows_per_thread + (i < remainder_rows ? 1 : 0);
                if (rows_for_this_thread == 0 && current_y_start >= basic_height) continue; // Avoid empty tasks
                unsigned int y_start = current_y_start;
                unsigned int y_end = current_y_start + rows_for_this_thread;
                if (y_start < y_end) { // Ensure valid range
                    render_targets.emplace_back(0, y_start, basic_width, y_end);
                    current_y_start = y_end;
                } else if (current_y_start >= basic_height) {
                    break; // Stop if all rows are assigned
                }
            }

            unsigned int actual_threads_to_launch = render_targets.size();

            // --- Thread Limit Check ---
            // Ensure we don't try to launch more threads than we have control flags for.
            if (actual_threads_to_launch > thread_stop_flags.size()) {
                std::cerr << "Warning: Clamping required threads (" << actual_threads_to_launch
                          << ") to available flags (" << thread_stop_flags.size() << ")." << std::endl;
                actual_threads_to_launch = thread_stop_flags.size();
                // This could result in parts of the image not being rendered.
            }

            if (actual_threads_to_launch == 0 && basic_height > 0 && basic_width > 0) {
                // Edge case: image has dimensions, but no threads assigned (e.g., height too small or clamped).
                return; // Exit lambda.
            }

            // --- Launch Worker Threads ---
            for (unsigned int i = 0; i < actual_threads_to_launch; ++i) {
                // Set flag state to '0' (working).
                // memory_order_release: Make this write visible before the thread potentially reads it.
                thread_stop_flags[i].store(0, std::memory_order_release);

                // Launch the actual rendering function (e.g., cpu_render_mandelbrot) in a new thread.
                std::thread t(cpu_render_mandelbrot, render_targets[i], pixels, basic_width, basic_height,
                              zoom_x, zoom_y, x_offset, y_offset, palette.data(), paletteSize,
                              max_iterations, h_total_iterations, std::ref(thread_stop_flags[i])); // Pass flag by ref

                // Detach the worker thread: The main_thread won't wait (join) for it directly.
                // Completion is tracked using the atomic stop flags.
                t.detach();
            }

            // --- Wait for Completion & Intermediate Updates ---
            // This loop polls the status flags of the worker threads.
            while (true) {
                bool all_done = true;
                if (actual_threads_to_launch > 0) {
                    for(unsigned int i = 0; i < actual_threads_to_launch; ++i) {
                        // Check if the flag is '1' (finished).
                        // memory_order_acquire: Ensures reads here see the final writes from the worker.
                        if (thread_stop_flags[i].load(std::memory_order_acquire) != 1) {
                            all_done = false;
                            break;
                        }
                    }
                }

                if (all_done) {
                    break; // All workers finished.
                }

                // Allows updating the display or performing other tasks periodically while rendering.
                post_processing();
                // Sleep briefly to avoid pegging the CPU core running this management thread.
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

            // Final update after all threads are confirmed done.
            post_processing();

        }); // End of lambda for main_thread

        // Detach the management thread itself. This makes the entire CPU render operation
        // asynchronous from the perspective of the function caller. The OS handles the detached thread.
        // The RenderGuard inside the lambda ensures the global lock is eventually released.
        main_thread.detach();
        return;

    } // End of if (!isCudaAvailable)
    RECALCULATE_RENDER_PARAMS(quality);
    if(!custom_formula){
        if (render_zoom_x > 1e7) {
            set_grid({10, 10});
            fractal_rendering<double><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                            x_offset, y_offset, d_palette, paletteSize,
                            max_iterations, d_total_iterations
            );
        }
        else {
            set_grid({32, 32});

            fractal_rendering<float><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                            x_offset, y_offset, d_palette, paletteSize,
                            max_iterations, d_total_iterations
            );
        }
    }
    else { // Custom, Formula handling
        cuCtxSetCurrent(ctx);

        double render_zoom_x_d = render_zoom_x;
        double render_zoom_y_d = render_zoom_y;
        double x_offset_d = x_offset;
        double y_offset_d = y_offset;
        float max_iterations_val = max_iterations;
        double max_iterations_d = max_iterations;
        unsigned int paletteSize_val = paletteSize;
        unsigned int width_val = width;
        unsigned int height_val = height;

        if (zoom_x > 1e7) {
            set_grid({10, 10});
            void* args[] = {
                    &cu_d_pixels,
                    &len,
                    &width_val,
                    &height_val,
                    &render_zoom_x_d,
                    &render_zoom_y_d,
                    &x_offset_d,
                    &y_offset_d,
                    &cu_palette,
                    &paletteSize_val,
                    &max_iterations_d,
                    &cu_d_total_iterations
            };
            CU_SAFE_CALL(cuLaunchKernel(kernelDouble, dimGrid.x, dimGrid.y, 1,
                                        dimBlock.x, dimBlock.y, 1,
                                        0, CUss,
                                        args, nullptr));
        } else { // Запуск float-версии
            set_grid({32, 32});
            float render_zoom_x_f = static_cast<float>(render_zoom_x_d);
            float render_zoom_y_f = static_cast<float>(render_zoom_y_d);
            float x_offset_f = static_cast<float>(x_offset_d);
            float y_offset_f = static_cast<float>(y_offset_d);

            void* args[] = {
                    &cu_d_pixels,
                    &len,
                    &width_val,
                    &height_val,
                    &render_zoom_x_f,
                    &render_zoom_y_f,
                    &x_offset_f,
                    &y_offset_f,
                    &cu_palette,
                    &paletteSize_val,
                    &max_iterations_val,
                    &cu_d_total_iterations
            };
            CU_SAFE_CALL(cuLaunchKernel(kernelFloat,
                                        dimGrid.x, dimGrid.y, 1,
                                        dimBlock.x, dimBlock.y, 1,
                                        0, CUss,
                                        args, nullptr));
        }

    }


    cudaError_t err = cudaGetLastError();
    while (err != cudaSuccess && context != context_type::NVRTC) {
        set_grid({dimBlock.x - 2, dimBlock.y - 2});
        if (zoom_x > 1e7) {
            fractal_rendering<double><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                            x_offset, y_offset, d_palette, paletteSize,
                            max_iterations, d_total_iterations
            );
        }
        else {
            fractal_rendering<float><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                            x_offset, y_offset, d_palette, paletteSize,
                            max_iterations, d_total_iterations
            );
        }
        err = cudaGetLastError();
        if (dimBlock.x < 3) {
            std::cerr << "Critical Issue (mandelbrot set): " << cudaGetErrorString(err) << "\n";
            return;
        }
    }
    ++counter;
    post_processing();
}