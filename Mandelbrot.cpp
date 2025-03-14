#include <SFML/Graphics.hpp>
#include <complex>
#include <cmath>
#include <vector_alias.hpp>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <thread>

unsigned int max_iterations = 50;

double basic_zoom_x = 240.0f;
double basic_zoom_y = 240.0f;

double zoom_x = basic_zoom_x;
double zoom_y = basic_zoom_y;

double x_offset = 3.5f;
double y_offset = 2.5f;

double zoom_factor = 1.0f;
double zoom_speed = 0.1f;

sf::Vector2i drag_start_pos;
bool is_dragging = false;


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



void Gradient(double& current_iteration, std::complex<double> complex, double& max_iter) {
	if (current_iteration >= max_iter) return;

	current_iteration = std::sqrt(current_iteration);
	max_iter = std::sqrt(max_iter);
}


void HSVtoRGB(double h, double s, double v, unsigned int& r, unsigned int& g, unsigned int& b) {
	h = fmod(h, 360.0);
	double c = v * s;
	double x = c * (1 - std::fabs(fmod(h / 60.0, 2) - 1));
	double m = v - c;

	double r_, g_, b_;
	if (h < 60) { r_ = c, g_ = x, b_ = 0; }
	else if (h < 120) { r_ = x, g_ = c, b_ = 0; }
	else if (h < 180) { r_ = 0, g_ = c, b_ = x; }
	else if (h < 240) { r_ = 0, g_ = x, b_ = c; }
	else if (h < 300) { r_ = x, g_ = 0, b_ = c; }
	else { r_ = c, g_ = 0, b_ = x; }

	r = static_cast<unsigned int>((r_ + m) * 255);
	g = static_cast<unsigned int>((g_ + m) * 255);
	b = static_cast<unsigned int>((b_ + m) * 255);
}



raw::vector<sf::Color> createHSVPalette(int numColors) {
	raw::vector<sf::Color> palette;
	for (int i = 0; i < numColors; ++i) {
		double t = static_cast<double>(i) / numColors;

		double hue;
		double saturation = 1.0;
		double value;

		if (t < 0.2) {
			hue = 240.0;
			value = 0.3 + t * (1.0 - 0.3) / 0.2;
		}
		else {
			hue = std::pow((t - 0.2) / 0.8, 0.5) * 360.0;
			value = 1.0;
		}

		unsigned int r, g, b;
		HSVtoRGB(hue, saturation, value, r, g, b);
		palette.push_back(sf::Color(r, g, b, 255));
	}
	return palette;
}
raw::vector<sf::Color> palette = createHSVPalette(200000);


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
	if (current_iteration == max_iterations) {
		return current_iteration;
	}
	return (current_iteration + 1 - std::log2(std::log2(std::abs(z))));
}


double Gradient(double current_iteration, std::complex<double> complex) {
	if (current_iteration >= max_iterations) return 0.0;
	return (current_iteration) / static_cast<double>(max_iterations);
}


void Render_Section(double& max_iters, sf::Image& Non_compressed, unsigned int start_x = 0, unsigned int end_x = 1600) {
	for (unsigned int x = start_x; x < end_x; x++) {
		for (unsigned int y = 0; y < 1200; y++) {
			double real = x / zoom_x - x_offset;
			double imag = y / zoom_y - y_offset;
			double iter = mandelbrot(real, imag);
			double gradient = Gradient(iter, std::complex<double>(real, imag));
			Gradient(iter, std::complex<double>(real, imag), max_iters);
			if (iter >= max_iterations) {
				Non_compressed.setPixel({ x, y }, sf::Color::Black);
			}
			else {
				int index = static_cast<int>(gradient * (palette.get_size() - 1));
				index = std::clamp(index, 0, static_cast<int>(palette.get_size() - 1));
				Non_compressed.setPixel({ x, y }, palette[index]);
			}
		}
	}
}



static sf::Image ANTIALIASING_SSAA4(sf::Image& high_qual, unsigned int size) {
	sf::Image low_qual({ high_qual.getSize().x / size * 2, high_qual.getSize().y / size * 2 }, sf::Color::Black);
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



void handleZoom(float wheel_delta, const sf::Vector2i mouse_pos) {
	double old_zoom_x = zoom_x;
	double old_zoom_y = zoom_y;
	double old_x_offset = x_offset;
	double old_y_offset = y_offset;

	double zoom_change = 1.0 + wheel_delta * zoom_speed;
	zoom_factor *= zoom_change;

	zoom_factor = std::max(std::min(zoom_factor, 1000.0), 0.01);

	zoom_x = basic_zoom_x * zoom_factor;
	zoom_y = basic_zoom_y * zoom_factor;

	double image_mouse_x = mouse_pos.x * (1600.0 / 800.0);
	double image_mouse_y = mouse_pos.y * (1200.0 / 600.0);

	x_offset = old_x_offset + (image_mouse_x / zoom_x - image_mouse_x / old_zoom_x);
	y_offset = old_y_offset + (image_mouse_y / zoom_y - image_mouse_y / old_zoom_y);

	std::cout << "Zoom Factor: " << zoom_factor << ", Offset: (" << x_offset << ", " << y_offset << ")" << std::endl;
}



void start_dragging(sf::Vector2i mouse_pos) {
	is_dragging = true;
	drag_start_pos = mouse_pos;
}

void dragging(sf::Vector2i mouse_pos) {
	if (!is_dragging) return;

	sf::Vector2i delta_pos = mouse_pos - drag_start_pos;

	double delta_real = static_cast<double>(delta_pos.x) / zoom_x;
	double delta_imag = static_cast<double>(delta_pos.y) / zoom_y;

	x_offset += delta_real;
	y_offset += delta_imag;

	drag_start_pos = mouse_pos;
}

void stop_dragging() {
	is_dragging = false;
}




int main() {
	bool need_render = true;
	sf::RenderWindow window(sf::VideoMode({ 800, 600 }), "Mandelbrot");
	sf::Image image({ 1600, 1200 }, sf::Color::Black);
	sf::Image compressed({ 800, 600 }, sf::Color::Black);
	sf::Clock timer;

	double max_iters = max_iterations;

	double iter;
	unsigned int red;
	unsigned int green;
	unsigned int blue;
	
	sf::Vector2i mouse;

	while (window.isOpen()) {

		std::vector<std::thread> threads;
		unsigned int num_threads = std::thread::hardware_concurrency();
		unsigned int section_width = 1600 / num_threads;

		while (const auto event = window.pollEvent()) {

			if(const auto* mm = event->getIf<sf::Event::MouseMoved>()) mouse = mm->position;

			if (event->is<sf::Event::Closed>()) {
				window.close();
			}

			if (const auto* button = event->getIf<sf::Event::KeyPressed>()) {
				if (button->scancode == sf::Keyboard::Scancode::Escape) {
					window.close();
				}
			}

			if (const auto* mouseWheelScrolled = event->getIf<sf::Event::MouseWheelScrolled>()) {
				need_render = true;
				handleZoom(mouseWheelScrolled->delta, mouse);
			}


			if (const auto* mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
				if (mouseButtonPressed->button == sf::Mouse::Button::Left) {
					start_dragging(mouse);
				}
			}

			if (const auto* mouseButtonReleased = event->getIf<sf::Event::MouseButtonReleased>()) {
				if (mouseButtonReleased->button == sf::Mouse::Button::Left) {
					stop_dragging();
				}
			}

			if (const auto* mouseButtonReleased = event->getIf<sf::Event::MouseMoved>()) {
				if (is_dragging) need_render = true;
				dragging({mouse.x, mouse.y});
			}

		}
		for (unsigned int i = 0; i < num_threads; ++i) {
			unsigned int start_x = i * section_width;
			unsigned int end_x = (i == num_threads - 1) ? 1600 : (i + 1) * section_width;
			threads.emplace_back(Render_Section, std::ref(max_iters), std::ref(image), start_x, end_x);
		}

		for (auto& thread : threads) {
			thread.join();
		}
		//sf::Texture tt1(image);
		//sf::Sprite sprite1(tt1);
		//window.draw(sprite1, sprite1.getTransform());
		//window.display();
		if (need_render) {
			need_render = false;
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
}