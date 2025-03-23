#include "CUDA files/FractalClass.cuh"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>

void main_thread() {
    render_state curr_qual = render_state::good;
    render_state prev_qual = render_state::good;

   

    sf::RenderWindow window(sf::VideoMode({ 1920, 1080 }), "Fractals");

    double zx, zy;

    bool drawen = false;

    sf::Clock timer;
    sf::Clock timer_julia;

    bool is_dragging = false;

	bool julia_render = false;

    FractalBase<fractals::mandelbrot> mandelbrot;
    FractalBase<fractals::julia> julia_set;
    double max_iters = mandelbrot.get_max_iters();

    bool block_julia = false;

    bool mouse_moved = true;

    sf::Vector2i mouse;
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (const auto* mm = event->getIf<sf::Event::MouseMoved>()) {
                if(mouse.x < 800 && mouse.y < 600)
					mouse_moved = true;
                mouse = mm->position;
            } 

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
                else if(button->scancode == sf::Keyboard::Scancode::S) {
                    if (mouse.x < 800 && mouse.y < 600)
                        block_julia = !block_julia;
				}
            }
            
            if (const auto* mouseWheelScrolled = event->getIf<sf::Event::MouseWheelScrolled>()) {
                if (mouse.x < 800 && mouse.y < 600) {
                    curr_qual = render_state::good;
                    mandelbrot.handleZoom(mouseWheelScrolled->delta, mouse);
                }
				else if (mouse.x > window.getSize().x - 800 && mouse.y < 600) {
					julia_set.handleZoom(mouseWheelScrolled->delta, mouse);
                    julia_render = true;
				}
            }

            if (const auto* mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseButtonPressed->button == sf::Mouse::Button::Left) {
                    if (mouse.x < 800 && mouse.y < 600) {
                        mouse_moved = true;
                        curr_qual = render_state::good;
                        mandelbrot.start_dragging(mouse);
                    }
					else if (mouse.x > window.getSize().x - 800 && mouse.y < 600) {
						julia_set.start_dragging(mouse);
                        julia_render = true;
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
					if (julia_set.get_is_dragging()) {
						julia_set.dragging({ mouse.x, mouse.y });
                        julia_render = true;
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

        if(mouse_moved && !block_julia) {
            drawen = true;
            zx = -(mandelbrot.get_x_offset() - (mouse.x / mandelbrot.get_zoom_x()));

            zy = -(mandelbrot.get_y_offset() - (mouse.y / mandelbrot.get_zoom_y()));
             
            julia_set.render(render_state::good, zx, zy);


            julia_set.setPosition({ float(window.getSize().x - 800), 0 });

			window.clear();

            window.draw(julia_set);
            window.draw(mandelbrot);
			

			window.display();

            auto time = timer_julia.restart();

			mouse_moved = false;
		}

        else if (julia_render) {
			julia_set.render(render_state::best, zx, zy);
            julia_set.setPosition({ float(window.getSize().x - 800), 0 });

            window.clear();

            window.draw(julia_set);
            window.draw(mandelbrot);


            window.display();

			julia_render = false;
        }

    }
}



int main() {
    std::thread mainThread(main_thread);
    mainThread.join();
}