#include <TGUI/Backend/SFML-Graphics.hpp>
#include "CUDA files/FractalClass.cuh"
#include <iostream>
#include <thread>

void main_thread() {
    render_state curr_qual = render_state::bad;
    render_state prev_qual = render_state::bad;
    render_state prev_qual_jul = render_state::good;
    render_state curr_qual_jul = render_state::good;

   

    sf::RenderWindow window(sf::VideoMode({ 1920, 1080 }), "Fractals");
    sf::RenderTexture buffer({1920, 1080});

    sf::Font font;
    bool state = font.openFromFile("C:\\Windows\\Fonts\\ariblk.ttf");
    if (!state)
        return;
    sf::Text text(font);
    text.setString("0");


    sf::Vector2i base_mouse_pos;

    double zx, zy;

    bool drawen = false;
    bool pressed = true;
    bool first_move = true;

    sf::Clock timer;
    sf::Clock timer_julia;
    sf::Clock fps_clock;

    float fps = 0;

    bool is_dragging = false;

	bool julia_render = false;

    sf::Clock timer_to_better_qual;

    FractalBase<fractals::mandelbrot> mandelbrot;
    FractalBase<fractals::julia> julia_set;
    julia_set.setPosition({float(window.getSize().x - 800), 0.f});
    double max_iters = mandelbrot.get_max_iters();

    bool block_julia = false;
    
    bool mouse_moved = true;

    window.setFramerateLimit(360);

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
                else if(button->scancode == sf::Keyboard::Scancode::B) {
                    if (mouse.x < 800 && mouse.y < 600)
                        block_julia = !block_julia;
				}
                else if (button->scancode == sf::Keyboard::Scancode::W) {
                    if (mouse.x < 800 && mouse.y < 600) {
                        mouse_moved = true;
                        curr_qual = render_state::good;
                        mandelbrot.move_fractal({ 0, 1 });
                    }
                }
                else if (button->scancode == sf::Keyboard::Scancode::S) {
					if (mouse.x < 800 && mouse.y < 600) {
						mouse_moved = true;
						curr_qual = render_state::good;
						mandelbrot.move_fractal({ 0, -1 });
					}
                }
                else if (button->scancode == sf::Keyboard::Scancode::A) {
					if (mouse.x < 800 && mouse.y < 600) {
						mouse_moved = true;
						curr_qual = render_state::good;
						mandelbrot.move_fractal({ 1, 0 });
					}
                }
				else if (button->scancode == sf::Keyboard::Scancode::D) {
					if (mouse.x < 800 && mouse.y < 600) {
						mouse_moved = true;
						curr_qual = render_state::good;
						mandelbrot.move_fractal({ -1, 0 });
					}
				}
            }
            
            if (const auto* mouseWheelScrolled = event->getIf<sf::Event::MouseWheelScrolled>()) {
                if (mouse.x < 800 && mouse.y < 600) {
                    curr_qual = render_state::good;
                    mandelbrot.handleZoom(mouseWheelScrolled->delta, mouse);
                }
				else if (mouse.x > window.getSize().x - 800 && mouse.y < 600) {
					julia_set.handleZoom(mouseWheelScrolled->delta, mouse);
                    curr_qual_jul = render_state::good;
				}
            }

            if (const auto* mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                
                if (mouseButtonPressed->button == sf::Mouse::Button::Left) {
                    if (mouse.x < 800 && mouse.y < 600) {
                        pressed = true;
                        mouse_moved = true;
                        curr_qual = render_state::good;
                        mandelbrot.start_dragging(mouse);
                    }
					else if (mouse.x > window.getSize().x - 800 && mouse.y < 600) {
						julia_set.start_dragging(mouse);
                        curr_qual_jul = render_state::good;
					}
                }
            }

            if (const auto* mouseButtonReleased = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseButtonReleased->button == sf::Mouse::Button::Left) {
                    if (mouse.x < 800 && mouse.y < 600) {
                        pressed = false;
                        mandelbrot.stop_dragging();
                    }
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
                        curr_qual_jul = render_state::good;
					}
				}
            }
        }

        ++fps;


        if (curr_qual != render_state::best || prev_qual != render_state::best && !pressed) {
			prev_qual = curr_qual;
            std::string quality = curr_qual == render_state::best ? "Best" : curr_qual == render_state::good ? "Good" : "Bad";
            mandelbrot.render(curr_qual);

            auto startdrawing = std::chrono::high_resolution_clock::now();
            buffer.clear();
            window.clear();
            buffer.draw(mandelbrot);
            if (drawen) {
                buffer.draw(julia_set);
            }

            buffer.display();
            window.draw(sf::Sprite(buffer.getTexture()));
            window.draw(text);

            auto time = timer.restart();;
            window.display();
            auto enddrawing = std::chrono::high_resolution_clock::now();
            auto diffdrawing = std::chrono::duration_cast<std::chrono::milliseconds>(enddrawing - startdrawing);
            std::cout << "Time needed to apply SSAA4 and draw to window: " << diffdrawing.count() << "\n";
            std::cout << "Mandelbrot set " << "(" << quality << ")" << " was drew in : " << time.asMilliseconds() << std::endl;
            std::cout << "Zoom scale: " << mandelbrot.get_zoom_x() << "\n";
            if (timer_to_better_qual.getElapsedTime().asMilliseconds() > 500 && !pressed) {
                if (quality == "Bad") {
                    curr_qual = render_state::good;
                }

                if (quality == "Good") {
                    curr_qual = render_state::best;
                }
                timer_to_better_qual.restart();
            }
        }

        if(mouse_moved && !block_julia) {
            drawen = true;
            zx = -(mandelbrot.get_x_offset() - (mouse.x / mandelbrot.get_zoom_x()));

            zy = -(mandelbrot.get_y_offset() - (mouse.y / mandelbrot.get_zoom_y()));
             
            julia_set.render(render_state::good, zx, zy);


            julia_set.setPosition({ float(window.getSize().x - 800), 0 });

            buffer.clear();
            window.clear();

            buffer.draw(mandelbrot);
            buffer.draw(julia_set);

            buffer.display();

            window.draw(sf::Sprite(buffer.getTexture()));
            window.draw(text);

            window.display();

            auto time = timer_julia.restart();
            

			mouse_moved = false;
		}

        else if (curr_qual_jul != render_state::best || prev_qual_jul != render_state::best) {
            prev_qual_jul = curr_qual_jul;
            std::string quality = curr_qual_jul == render_state::best ? "Best" : "Good";

			julia_set.render(curr_qual_jul, zx, zy);
            julia_set.setPosition({ float(window.getSize().x - 800), 0 });

            buffer.clear();
            window.clear();
            buffer.draw(mandelbrot);
            if (drawen) {
                buffer.draw(julia_set);
            }

            buffer.display();
            window.draw(sf::Sprite(buffer.getTexture()));
            window.draw(text);
            window.display();
            

            std::cout << "julia set " << "(" << quality << ")\n";

            if (quality == "Good") {
                curr_qual_jul = render_state::best;
            }
        }
        else {
			window.clear();
            window.draw(sf::Sprite(buffer.getTexture()));
            window.draw(text);
            window.display();
        }

        if(fps_clock.getElapsedTime().asSeconds() > 0.2f) {
            float elapsed = fps_clock.getElapsedTime().asSeconds();
            text.setString(std::to_string(fps / elapsed));
            text.setPosition({ 10, 10 });
			fps = 0;
			fps_clock.restart();
		}

    }
}



int main() {
    std::thread mainThread(main_thread);
    mainThread.join();
}