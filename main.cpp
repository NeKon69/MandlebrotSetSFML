#include "CUDA files/FractalClass.cuh"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>

void main_thread() {
    render_state curr_qual = render_state::good;
    render_state prev_qual = render_state::good;

   

    sf::RenderWindow window(sf::VideoMode({ 1920, 1080 }), "Fractals");

    bool drawen = false;

    sf::Clock timer;
    sf::Clock timer_julia;

    bool is_dragging = false;

    FractalBase<fractals::mandelbrot> mandelbrot;
    FractalBase<fractals::julia> julia_set;
    double max_iters = mandelbrot.get_max_iters();

    bool mouse_moved = true;

    sf::Vector2i mouse;
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (const auto* mm = event->getIf<sf::Event::MouseMoved>()) mouse = mm->position;

            if (event->is<sf::Event::Closed>()) {
                window.close();
            }

            if (const auto* button = event->getIf<sf::Event::KeyPressed>()) {
                if (button->scancode == sf::Keyboard::Scancode::Escape) {
                    window.close();
                }
                else if (button->scancode == sf::Keyboard::Scancode::R) {
                    if (mouse.x < 800 && mouse.y < 600)
                        mandelbrot.set_max_iters(mandelbrot.get_max_iters() * 2);
                }
            }
            
            if (const auto* mouseWheelScrolled = event->getIf<sf::Event::MouseWheelScrolled>()) {
                if (mouse.x < 800 && mouse.y < 600) {
                    curr_qual = render_state::good;
                    mandelbrot.handleZoom(mouseWheelScrolled->delta, mouse);
                }
				else if (mouse.x > window.getSize().x - 800 && mouse.y < 600) {
					julia_set.handleZoom(mouseWheelScrolled->delta, mouse);
                    mouse_moved = true;
				}
            }

            if (const auto* mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseButtonPressed->button == sf::Mouse::Button::Left) {
                    if (mouse.x < 800 && mouse.y < 600) {
                        curr_qual = render_state::good;
                        mandelbrot.start_dragging(mouse);
                    }
					else if (mouse.x > window.getSize().x - 800 && mouse.y < 600) {
						julia_set.start_dragging(mouse);
                        mouse_moved = true;
					}
                }
            }

            if (const auto* mouseButtonReleased = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseButtonReleased->button == sf::Mouse::Button::Left) {
                    if (mouse.x < 800 && mouse.y < 600)
                        mandelbrot.stop_dragging();
                    else if (mouse.x > window.getSize().x - 800 && mouse.y < 600) {
                        julia_set.stop_dragging();
                    }
                }
            }

            if (const auto* mouseButtonReleased = event->getIf<sf::Event::MouseMoved>()) {
                if (mouse.x < 800 && mouse.y < 600) {
                    mouse_moved = true;
                    if (mandelbrot.get_is_dragging()) {
                        curr_qual = render_state::good;
                        mandelbrot.dragging({ mouse.x, mouse.y });
                    }
                }
                else if(mouse.x > window.getSize().x - 800 && mouse.y < 600 && julia_set.get_is_dragging()) {
                    mouse_moved = true;
					if (julia_set.get_is_dragging()) {
						julia_set.dragging({ mouse.x, mouse.y });
					}
				}
            }
        }

        if (curr_qual != render_state::best || prev_qual != render_state::best) {
			prev_qual = curr_qual;
            std::string quality = curr_qual == render_state::best ? "Best" : "Good";

            mandelbrot.render(curr_qual);
            window.clear();

            window.draw(mandelbrot);
            if(drawen)
                window.draw(julia_set);

            auto time = timer.restart();
			
            std::cout << "Mandelbrot set " << "(" << quality << ")" <<  " was drew in : " << time.asMilliseconds() << std::endl;

			if (quality == "Good") {
				curr_qual = render_state::best;
			}

            window.display();
        }

        if(mouse_moved) {
            std::cout << "Mouse pos: " << mouse.x << ", " << mouse.y << std::endl;
            drawen = true;

            double mandel_mouse_x = mouse.x; 
            double mandel_mouse_y = mouse.y; 

            double cx = mandelbrot.get_x_offset() + (mandel_mouse_x / 800.0) * (4.0 / (mandelbrot.get_zoom_x() * mandelbrot.get_zoom_scale()));
            double cy = mandelbrot.get_y_offset() + (mandel_mouse_y / 600.0) * (3.0 / (mandelbrot.get_zoom_y() * mandelbrot.get_zoom_scale()));

            julia_set.render(render_state::good, mandelbrot.get_x_offset(), mandelbrot.get_y_offset(),
                mandelbrot.get_zoom_x(), mandelbrot.get_zoom_y(),
                cx, cy);

			julia_set.render(render_state::good, mandelbrot.get_x_offset(), mandelbrot.get_y_offset(), 
                             mandelbrot.get_zoom_x(), mandelbrot.get_zoom_y(), 
                            ((mouse.x / 800.0) * 4.0) - 2.0, ((mouse.y / 600.0) * 3.0) - 1.5);

            julia_set.setPosition({ float(window.getSize().x - 800), 0 });

			window.clear();

            window.draw(julia_set);
            window.draw(mandelbrot);
			

			window.display();

            auto time = timer_julia.restart();

            std::cout << "Julia set " << "(" << "Best" << ")" << " was drew in : " << time.asMilliseconds() << std::endl;

			mouse_moved = false;
		}
    }
}



int main() {
    std::thread mainThread(main_thread);
    mainThread.join();
}