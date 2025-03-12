#include <SFML/Graphics.hpp>
#include <complex>
#include <cmath>
#include <vector_alias.hpp>
#include <unordered_map>
#include <random>

constexpr static unsigned int max_iterations = 150;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double mandelbrot(double x, double y) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_real_distribution<double> dist(-0.5, 0.5);
	std::complex<double> z = { 0.0, 0.0 };
	std::complex<double> c = { x, y };
	unsigned int current_iteration = 0;
	while (std::norm(z) < 4 && current_iteration < max_iterations) {
		z = z * z + c;
		current_iteration++;
	}
	if (std::abs(z) < 1.0) {
		return current_iteration;
	}
	return (current_iteration + 1 - std::log2(std::log2(std::abs(z))));
}

double Gradient(double current_iteration, std::complex<double> complex) {
	if (current_iteration >= max_iterations) return 0.0;
	return ((current_iteration + 1 - std::log2(std::log2(std::abs(complex)))) / max_iterations);
}

void HSVtoRGB(double h, double s, double v, unsigned int& r, unsigned int& g, unsigned int& b) {
	h = fmod(h, 360.0);
	double c = v * s;
	double x = c * (1 - std::fabs(fmod(h / 60.0, 2) - 1));
	double m = v - c;

	double r_, g_, b_;
	if (h < 60)		  { r_ = c, g_ = x, b_ = 0; }
	else if (h < 120) { r_ = x, g_ = c, b_ = 0; }
	else if (h < 180) { r_ = 0, g_ = c, b_ = x; }
	else if (h < 240) { r_ = 0, g_ = x, b_ = c; }
	else if (h < 300) { r_ = x, g_ = 0, b_ = c; }
	else			  { r_ = c, g_ = 0, b_ = x; }

	r = static_cast<unsigned int>((r_ + m) * 255);
	g = static_cast<unsigned int>((g_ + m) * 255);
	b = static_cast<unsigned int>((b_ + m) * 255);
}

static sf::Image ANTIALIASING_SSAA4(sf::Image& high_qual, unsigned int size) {
	sf::Image low_qual({high_qual.getSize().x / size * 2, high_qual.getSize().y / size * 2}, sf::Color::Black);
	for (unsigned int x = 0; x < high_qual.getSize().x; x += 2) {
		for (unsigned int y = 0; y < high_qual.getSize().y; y += 2) {
			sf::Color colorTopLeft = high_qual.getPixel({ x, y });
			sf::Color colorTopRight = high_qual.getPixel({ x + 1, y });
			sf::Color colorBottomLeft = high_qual.getPixel({ x, y + 1 });
			sf::Color colorBottomRight = high_qual.getPixel({ x + 1, y + 1 });
			sf::Color color = sf::Color((colorTopLeft.r + colorTopRight.r + colorBottomLeft.r + colorBottomRight.r) / 4, (colorTopLeft.g + colorTopRight.g + colorBottomLeft.g + colorBottomRight.g) / 4, (colorTopLeft.b + colorTopRight.b + colorBottomLeft.b + colorBottomRight.b) / 4);
			low_qual.setPixel({ x / 2, y / 2 }, color);
		}
	}
	return low_qual;
}

int main() {
	sf::RenderWindow window(sf::VideoMode({ 800, 600 }), "Mandelbrot");
	sf::Image image({ 1600, 1200 }, sf::Color::Black);
	sf::Image compressed({ 800, 600 }, sf::Color::Black);
	sf::Clock timer;
	double iter;
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
		for (unsigned int x = 0; x < 1600; x++) {
			for (unsigned int y = 0; y < 1200; y++) {
				double real = x / 240.0 - 3.5f;
				double imag = y / 240.0 - 2.5f;
				iter = mandelbrot(real, imag);
				double gradient = Gradient(iter, std::complex<double>(real, imag));
				double hue = 360.0 * (0.5 + 0.5 * std::sin(2 * M_PI * gradient));
				HSVtoRGB(hue, 1.0, iter < max_iterations ? 1.0 : 0.0, red, green, blue);
				image.setPixel({ x, y }, sf::Color(red, green, blue, 255));
			}
		}
		//sf::Texture tt1(image);
		//sf::Sprite sprite1(tt1);
		//window.draw(sprite1, sprite1.getTransform());
		//window.display();
		compressed = ANTIALIASING_SSAA4(image, 4);
		window.clear();
		sf::Texture tt(compressed);
		sf::Sprite sprite(tt);
		sprite.setPosition({ 0, 0 });
		window.draw(sprite, sprite.getTransform());
		auto time = timer.restart();
		std::cout << "Mandelbrot set was drew in: " << time.asMilliseconds() << std::endl;
		window.display();
	} 
}
