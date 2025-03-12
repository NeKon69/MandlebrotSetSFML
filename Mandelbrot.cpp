#include <SFML/Graphics.hpp>
#include <vector_alias.hpp>
#include <complex>

constexpr unsigned int max_iterations = 1000;

unsigned int mandelbrot(double x, double y) {
    std::complex<double> z = { 0.0, 0.0 };
    std::complex<double> c = { x, y };
    unsigned int current_iteration = 0;
    while (current_iteration < max_iterations && std::norm(z) < 4) {
        z = z * z + c;
        current_iteration++;
    }
    return current_iteration;
}

int main(){
	sf::RenderWindow window(sf::VideoMode({ 800, 600 }), "Mandelbrot");
	sf::Image image({ 800, 600 }, sf::Color::Black);
	unsigned int iter;
	unsigned int red;
	unsigned int green;
	unsigned int blue;
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
			if (event->is<sf::Event::Closed>()) {
				window.close();
			}
            if (const auto* button = event->getIf<sf::Event::KeyPressed>()) {
                if (button->scancode == sf::Keyboard::Scancode::Escape) {
					window.close();
                }
            }
        }
		window.clear();
		for (unsigned int x = 0; x < 800; x++) {
			for (unsigned int y = 0; y < 600; y++) {
				iter = mandelbrot(x, y);
				red = 255 - (iter % 255);
				green = 255 - (iter * 2 % 255);
				blue = 255 - (iter * 4 % 255);
				image.setPixel({ x, y }, sf::Color(red, green, blue, 255));
			}
		}
		sf::Texture tt(image);
		sf::Sprite sprite(tt);
		window.draw(sprite, sprite.getTransform());
		window.display();
    }

}