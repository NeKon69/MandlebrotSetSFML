#include "CUDA files/FractalClass.cuh"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>

template class FractalBase<fractals::mandelbrot>;
void main_thread() {
    render_state curr_qual = render_state::good;
    render_state prev_qual = render_state::good;
    sf::RenderWindow window(sf::VideoMode({ 800, 600 }), "Mandelbrot");
    sf::Image image({ 1600, 1200 }, sf::Color::Black);
    sf::Image compressed({ 800, 600 }, sf::Color::Black);
    sf::Clock timer;

    bool is_dragging = false;

    FractalBase<fractals::mandelbrot> mandelbrot;
    double max_iters = mandelbrot.get_max_iters();

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
            }

            if (const auto* mouseWheelScrolled = event->getIf<sf::Event::MouseWheelScrolled>()) {

                curr_qual = render_state::good;
                mandelbrot.handleZoom(mouseWheelScrolled->delta, mouse);
            }

            if (const auto* mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseButtonPressed->button == sf::Mouse::Button::Left) {
                    curr_qual = render_state::good;
                    mandelbrot.start_dragging(mouse);
                }
            }

            if (const auto* mouseButtonReleased = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseButtonReleased->button == sf::Mouse::Button::Left) {
                    mandelbrot.stop_dragging();
                }
            }

            if (const auto* mouseButtonReleased = event->getIf<sf::Event::MouseMoved>()) {
                if (mandelbrot.get_is_dragging()) {
                    curr_qual = render_state::good;
                    mandelbrot.dragging({ mouse.x, mouse.y });
                }
            }
        }

        if (curr_qual != render_state::best || prev_qual != render_state::best) {
			prev_qual = curr_qual;
            std::string quality = curr_qual == render_state::best ? "Best" : "Good";

            mandelbrot.render(curr_qual);

            window.draw(mandelbrot);

            auto time = timer.restart();
			
            std::cout << "Mandelbrot set " << "(" << quality << ")" <<  " was drew in : " << time.asMilliseconds() << std::endl;

			if (quality == "Good") {
				curr_qual = render_state::best;
			}

            window.display();
        }
    }
}



int main() {
    std::thread mainThread(main_thread);
    mainThread.join();
}