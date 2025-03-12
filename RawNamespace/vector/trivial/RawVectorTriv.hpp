#pragma once
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "RawNamespace\vector\RawVector.hpp"

namespace raw {
	template<typename T>
	class vector_triv : public vector_base<T> {
	private:

		/*************************************************************************************************
		 * PRIVATE MEMBERS:
		 *
		 * `data`     - Stores allocated memory for `capacity` elements of type `T`.
		 * `size`     - Number of currently used elements.
		 * `capacity` - Total allocated memory size for `data`.
		 *************************************************************************************************/

		 /// Data members data, size, capacity ARE ALREADY inherited from vector_base, no need to declare them again here!

		 /*************************************************************************************************
		  * PRIVATE FUNCTION: `normalize_capacity()`
		  *
		  * Reallocates `data` with doubled `capacity` if needed (when `size` >= `capacity`).
		  * Copies existing data to new memory.
		  * Returns pointer to the new `data`.
		  *
		  * Throws: std::bad_alloc on allocation failure.
		  *************************************************************************************************/
		T* normalize_capacity() override {
			while (size >= capacity) capacity *= 2;
			T* new_data = (T*)realloc(data, sizeof(T) * capacity);
			if (new_data) {
				data = new_data;
				return data;
			}
			throw std::bad_alloc();
		}

	protected:
		using vector_base<T>::data;
		using vector_base<T>::size;
		using vector_base<T>::capacity;

	public:
		using Iterator = typename vector_base<T>::template IteratorBase<T>;
		using const_iterator = typename vector_base<T>::template IteratorBase<const T>;

		/*********************************************************************
		 * DEFAULT CONSTRUCTOR: `vector_triv()`
		 *
		 * Initializes with capacity for 1 element.
		 *********************************************************************/
		vector_triv() {
			data = (T*)malloc(sizeof(T));
		};

		/*********************************************************************
		 * SIZE CONSTRUCTOR: `vector_triv(size_t size_)`
		 *
		 * Initializes with given `size_`.
		 * Reserves memory and initializes elements to 0.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 *********************************************************************/
		vector_triv(size_t size_) {
			size = size_;
			capacity = 1;
			try {
				normalize_capacity();
				std::memset(data, 0, size_ * sizeof(T));
			}
			catch (std::bad_alloc& e) {
				std::cerr << e.what() << std::endl;
				free(data);
				throw;
			}
		};

		/**************************************************************************************
		 * COPY CONSTRUCTOR: `vector_triv(const vector_triv& other)`
		 *
		 * Deep copy from `other`.
		 * Copies size, capacity, and data.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 **************************************************************************************/

		vector_triv(const vector_triv& other) {
			size = other.size;
			capacity = other.capacity;
			try {
				T* newData = (T*)calloc(capacity, sizeof(T));
				if (!newData) {
					throw std::bad_alloc();
				}
				std::memcpy(newData, other.data, size * sizeof(T));
				free(data);
				data = newData;
			}
			catch (std::bad_alloc& e) {
				std::cerr << e.what() << std::endl;
				throw;
			}
		}

		/*************************************************************************************
		 * MOVE CONSTRUCTOR: `vector_triv(vector_triv&& other) noexcept`
		 *
		 * Move resources from `other`.
		 * Transfers data, size, and capacity. `other` becomes empty.
		 *************************************************************************************/

		vector_triv(vector_triv&& other) noexcept {
			data = other.data;
			size = other.size;
			capacity = other.capacity;

			other.data = nullptr;
			other.size = 0;
			other.capacity = 0;
		}

		/*************************************************************************************
		 * COPY ASSIGNMENT: `operator=(const vector_triv& other)`
		 *
		 * Deep copy assignment from `other`.
		 * Frees current data, then copies size, capacity, and data from `other`.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 *************************************************************************************/

		vector_triv& operator=(const vector_triv& other) {
			if (this == &other) {
				return *this;
			}
			try {
				T* newData = (T*)calloc(other.capacity, sizeof(T));
				if (!newData) {
					throw std::bad_alloc();
				}
				std::memcpy(newData, other.data, other.size * sizeof(T));
				free(data);
				data = newData;
				size = other.size;
				capacity = other.capacity;
			}
			catch (std::bad_alloc& e) {
				std::cerr << e.what() << std::endl;
				throw;
			}
			return *this;
		}

		/*************************************************************************************
		 * MOVE ASSIGNMENT: `operator=(vector_triv&& other) noexcept`
		 *
		 * Move assignment from `other`.
		 * Frees current data, then moves data, size, and capacity from `other`.
		 * `other` becomes empty.
		 *************************************************************************************/

		vector_triv& operator=(vector_triv&& other) noexcept {
			if (this != &other) {
				free(data);
				data = other.data;
				size = other.size;
				capacity = other.capacity;
				other.data = nullptr;
				other.size = 0;
				other.capacity = 0;
			}
			return *this;
		}

		/*************************************************************************************
		 * PUSH_BACK FUNCTIONS: `push_back(const T& elem)`, `push_back(T&& elem)`
		 *
		 * Appends element `elem` to the end.
		 * Reallocates memory if `capacity` is full (using `normalize_capacity`).
		 * Overloads for copy and move semantics.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 *************************************************************************************/

		void push_back(const T& elem) override {
			if (!data)
				data = (T*)malloc(sizeof(T));
			try {
				if (capacity <= size) {
					capacity *= 2;
					auto newdata = (T*)std::realloc(data, capacity * sizeof(T));
					if (!newdata) {
						std::cerr << "Couldn't reserve that much space ERR" << std::endl;
						free(data);
						throw std::bad_alloc();
					}
					data = newdata;
				}
				if (data)
					data[size] = elem;
				++size;
			}
			catch (...) {
				std::cerr << "Something happened in push_back" << std::endl;
				throw;
			}
		}

		void push_back(T&& elem) override {
			if (!data)
				data = (T*)malloc(sizeof(T));
			try {
				if (capacity <= size) {
					normalize_capacity();
				}
				if (data) {
					data[size] = std::move(elem);
					++size;
				}
				else {
					throw std::bad_alloc();
				}
			}
			catch (...) {
				std::cerr << "Something happened in push_back" << std::endl;
				throw;
			}
		}

		/*********************************************************************
		 * INDEXING FUNCTIONS: `at()`, `operator[]` (const and non-const)
		 *
		 * Access elements by index.
		 * `at()` - with bounds checking, throws `std::out_of_range`.
		 * `operator[]` - no bounds checking.
		 *********************************************************************/

		T& at(const size_t index) override {
			if (index >= size || index < 0) {
				throw std::out_of_range("Index out of range");
			}
			return data[index];
		}

		const T& at(const size_t index) const override {
			if (index >= size || index < 0) {
				throw std::out_of_range("Index out of range");
			}
			return data[index];
		}

		T& operator [] (const size_t index) override {
			return data[index];
		}

		const T& operator [] (const size_t index) const override {
			return data[index];
		}

		/*********************************************************************
		 * GETTER FUNCTIONS: `get_size()`, `get_capacity()`
		 *
		 * Returns current `size` and `capacity`.
		 *********************************************************************/

		size_t get_size() const {
			return size;
		}
		size_t get_capacity() const {
			return capacity;
		}

		/*************************************************************************************
		 * RESIZE FUNCTION: `resize(size_t new_size)`
		 *
		 * Changes vector size to `new_size`.
		 * Reallocates if needed, initializes new elements to 0.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 *************************************************************************************/

		void resize(size_t new_size) override {
			if (new_size <= size) {
				size = new_size;
				return;
			}
			size_t old_size = size;
			size = new_size;
			normalize_capacity();
			if (old_size < size) { memset(data + old_size, 0, (size - old_size) * sizeof(T)); }

		}

		/*************************************************************************************
		 * RESERVE FUNCTION: `reserve(size_t reserve_size)`
		 *
		 * Reserves memory for at least `reserve_size` elements.
		 * Does not change `size`.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 *************************************************************************************/

		void reserve(size_t reserve_size) override {
			if (reserve_size <= capacity) {
				return;
			}
			try {
				auto temp = (T*)realloc(data, reserve_size * sizeof(T));
				if (!temp) {
					std::cerr << "Couldn't reserve that much space ERR" << std::endl;
					throw std::bad_alloc();
				}
				data = temp;
				capacity = reserve_size;
			}
			catch (std::bad_alloc& err) {
				std::cerr << err.what() << std::endl;
				throw;
			}
			catch (...) {
				std::cerr << "What tf happened here? P.S. -> (check reserve function)";
				throw;
			}
		}

		/*********************************************************************
		 * CLEAR FUNCTION: `clear()`
		 *
		 * Removes all elements, frees memory.
		 * Resets to initial state (capacity 1).
		 *********************************************************************/

		void clear() override {
			if (data) {
				free(data);
				data = nullptr;
			}
			size = 0;
			capacity = 1;
			data = (T*)malloc(sizeof(T));
		}

		/**************************************************************************************
		 * SHRINK_TO_FIT FUNCTION: `shrink_to_fit()`
		 *
		 * Reduces capacity to match `size`, minimizing memory.
		 * Can be inefficient if called frequently.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 **************************************************************************************/

		void shrink_to_fit() override {
			if (size != capacity && size != 0 && capacity != 1) {
				try {
					auto shrinked = (T*)realloc(data, size * sizeof(T));
					if (!shrinked) {
						free(data);
						data = nullptr;
						throw std::bad_alloc();
					}
					data = shrinked;
					capacity = size;
				}
				catch (std::bad_alloc& err) {
					std::cerr << err.what() << std::endl;
					throw;
				}
			}
			else { return; }

		}

		/*********************************************************************
		 * POP_BACK FUNCTION: `pop_back()`
		 *
		 * Removes the last element (decrements `size`).
		 * Does not actually deallocate memory.
		 *
		 * Throws: std::out_of_range if vector is empty.
		 *********************************************************************/

		void pop_back() override {
			if (size == 0) {
				throw std::out_of_range("Vector is empty");
			}
			if (--size == 0) { free(data); data = (T*)calloc(1, sizeof(T)); capacity = 1; }
		}

		/*************************************************************************************
		 * INSERT FUNCTIONS: `insert()` overloads
		 *
		 * Inserts element at given index/iterator.
		 * Shifts existing elements.
		 * Reallocates if needed.
		 * Overloads for copy and move semantics, and index/iterator position.
		 *
		 * Throws: std::out_of_range, std::bad_alloc.
		 *************************************************************************************/

		void insert(size_t index, const T& value) override {
			if (index > size) {
				throw std::out_of_range("Index out of range");
			}
			if (size == capacity) {
				normalize_capacity();
			}
			std::memmove(data + index + 1, data + index, (size - index) * sizeof(T));
			data[index] = value;
			++size;
		}

		void insert(size_t index, T&& value) override {
			if (index > size || index < 0) {
				throw std::out_of_range("Index out of range");
			}
			if (size == capacity) {
				normalize_capacity();
			}
			std::memmove(data + index + 1, data + index, (size - index) * sizeof(T));
			data[index] = std::move(value);
			++size;
		}

		Iterator insert(Iterator pos, const T& value) override {
			size_t insert_index = pos - Iterator(data);

			if (insert_index > size) {
				throw std::out_of_range("Index out of range");
			}
			if (size == capacity) {
				auto newData = normalize_capacity();
			}
			std::memmove(data + insert_index + 1, data + insert_index, (size - insert_index) * sizeof(T));
			data[insert_index] = value;
			++size;
			return Iterator(data + insert_index);
		}

		Iterator insert(Iterator pos, T&& value) override {
			size_t insert_index = pos - Iterator(data);

			if (insert_index > size) {
				throw std::out_of_range("Index out of range");
			}
			if (size == capacity) {
				auto newData = normalize_capacity();
			}
			std::memmove(data + insert_index + 1, data + insert_index, (size - insert_index) * sizeof(T));
			data[insert_index] = std::move(value);
			++size;
			return Iterator(data + insert_index);
		}

		/*********************************************************************
		 * ERASE FUNCTIONS: `erase()` overloads
		 *
		 * Erases element at given index/iterator.
		 * Shifts subsequent elements.
		 * Overloads for index and iterator position.
		 *
		 * Throws: std::out_of_range.
		 *********************************************************************/

		void erase(size_t index) override {
			if (index >= size || index < 0) {
				throw std::out_of_range("Index out of range");
			}
			std::memmove(data + index, data + index + 1, (size - index - 1) * sizeof(T));
			--size;
		}

		Iterator erase(Iterator pos) override {
			size_t erase_index = pos - Iterator(data);
			if (erase_index >= size || erase_index < 0) {
				throw std::out_of_range("Index out of range");
			}
			std::memmove(data + erase_index, data + erase_index + 1, (size - erase_index - 1) * sizeof(T));
			--size;
			return Iterator(data + erase_index);
		}

		/************************************************************
		 * SWAP FUNCTION: `swap(vector_base<T>& other) noexcept`
		 *
		 * Swaps data, size, and capacity with `other`.
		 * Fast, no memory operations.
		 *************************************************************/

		void swap(vector_base<T>& other) noexcept override {
			std::swap(data, other.data);
			std::swap(size, other.size);
			std::swap(capacity, other.capacity);
		}

		/*************************************************************************************
		 * COPY FUNCTION: `copy()`
		 *
		 * Returns a deep copy of the vector.
		 * New vector with copied data and capacity.
		 *************************************************************************************/

		vector_triv copy() {
			vector_triv<T> new_vector(size);
			new_vector.reserve(capacity);
			std::memcpy(new_vector.data, data, size * sizeof(T));
			return new_vector;
		}

		/// CHECKS IF OBJECT IS EMPTY, IF SIZE VARIABLE ISN'T HANDELED GOOD ENOUGH, CAN CAUSE CRUSH
		bool empty() { return size == 0; }

		/*********************************************************************
		 * DESTRUCTOR: `~vector_triv()`
		 *
		 * Prints destruction message (for debugging).
		 * Memory freeing handled in base class.
		 *********************************************************************/

		~vector_triv() override {
			std::cout << "vector_triv Object Destroyed next comes memory freeing" << std::endl;
		}
	};
}