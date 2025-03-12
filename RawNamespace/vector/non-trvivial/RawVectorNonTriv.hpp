#pragma once
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include "RawNamespace\vector\RawVector.hpp"


namespace raw {
	template<typename T>
	class vector_non_triv : public vector_base<T> {
	private:

		/*************************************************************************************
		 * PRIVATE FUNCTION: `normalize_capacity()`
		 *
		 * Reallocates `data` with doubled capacity if needed.
		 * Constructs new elements in new memory using move semantics.
		 * Destroys old elements.
		 * Returns pointer to the new `data`.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 *************************************************************************************/

		T* normalize_capacity() {
			while (size >= capacity) capacity *= 2;

			void* raw = malloc(sizeof(T) * capacity);
			if (!raw) throw std::bad_alloc();

			T* new_data = static_cast<T*>(raw);

			for (size_t i = 0; i < size; ++i) {
				new (new_data + i) T(std::move(data[i]));
				data[i].~T();
			}
			T* old_data = data;
			data = new_data;
			free(old_data);

			return new_data;
		}

		/*************************************************************************************
		 * PRIVATE FUNCTION: `normalize_capacity(size_t size_)`
		 *
		 * Reallocates `data` with doubled capacity to fit `size_`.
		 * Constructs new elements in new memory using move semantics.
		 * Destroys old elements.
		 * Returns pointer to the new `data`.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 *************************************************************************************/

		T* normalize_capacity(size_t size_) {
			if (std::numeric_limits<size_t>::max() / 2 < size_) throw std::bad_alloc();
			if (capacity == 0) capacity = 1;
			while (size_ >= capacity) capacity *= 2;

			void* raw = malloc(sizeof(T) * capacity);
			if (!raw) throw std::bad_alloc();

			T* new_data = static_cast<T*>(raw);

			for (size_t i = 0; i < size; ++i) {
				new (new_data + i) T(std::move(data[i]));
				data[i].~T();
			}
			T* old_data = data;
			data = new_data;
			free(old_data);

			return new_data;
		}

		/*************************************************************************************
		 * PRIVATE FUNCTION: `reserve_space()`
		 *
		 * Reserves memory for `capacity` elements.
		 * Constructs default-initialized elements in the reserved space.
		 * Returns pointer to the new `data`.
		 *
		 * Throws: std::bad_alloc on allocation failure.
		 *************************************************************************************/

		T* reserve_space() {
			while (size >= capacity) capacity *= 2;

			void* raw = malloc(sizeof(T) * capacity);
			if (!raw) throw std::bad_alloc();

			T* new_data = static_cast<T*>(raw);
			for (size_t i = 0; i < size; ++i)
				new (new_data + i) T();
			return new_data;
		}
	protected:
		using vector_base<T>::data;
		using vector_base<T>::size;
		using vector_base<T>::capacity;

	public:
		using Iterator = typename vector_base<T>::template IteratorBase<T>;
		using const_iterator = typename vector_base<T>::template IteratorBase<const T>;

		/*********************************************************************
		 * DEFAULT CONSTRUCTOR: `vector_non_triv()`
		 *
		 * Initializes with capacity 1 and default-constructs one element.
		 *
		 * Throws: std::bad_alloc on allocation failure, exceptions from T's constructor.
		 *********************************************************************/

		vector_non_triv() {
			size = 0;
			capacity = 1;
			T* raw_memory = (T*)malloc(sizeof(T));
			if (!raw_memory) throw std::bad_alloc();
			data = raw_memory;
		}

		/*********************************************************************
		 * SIZE CONSTRUCTOR: `vector_non_triv(size_t size_)`
		 *
		 * Initializes with given `size_`.
		 * Reserves memory and default-constructs `size_` elements.
		 *
		 * Throws: std::bad_alloc on allocation failure, exceptions from T's constructor.
		 *********************************************************************/

		vector_non_triv(size_t size_) {
			size = size_;
			capacity = 1;
			try {
				data = reserve_space();
			}
			catch (const std::bad_alloc& e) {
				if (data) {
					for (size_t i = 0; i < size; ++i)
						data[i].~T();
					free(data);
				}
				std::cerr << e.what() << std::endl;
				throw;
			}
		}

		/**************************************************************************************
		 * COPY CONSTRUCTOR: `vector_non_triv(const vector_non_triv& other)`
		 *
		 * Deep copy from `other`.
		 * Copies size, capacity, and constructs new elements by copying from `other`.
		 *
		 * Throws: std::bad_alloc on allocation failure, exceptions from T's copy constructor.
		 **************************************************************************************/

		vector_non_triv(const vector_non_triv& other) {
			size = other.size;
			capacity = other.capacity;
			try {
				data = (T*)malloc(sizeof(T) * capacity);
				for (size_t i = 0; i < size; ++i)
					new (data + i) T(other.data[i]);
			}
			catch (const std::bad_alloc& e) {
				if (data) {
					for (size_t i = 0; i < size; ++i)
						data[i].~T();
					free(data);
				}
				std::cerr << e.what() << std::endl;
				throw;
			}
		}

		/*************************************************************************************
		 * MOVE CONSTRUCTOR: `vector_non_triv(vector_non_triv&& other) noexcept`
		 *
		 * Move resources from `other`.
		 * Transfers size, capacity, and move-constructs elements from `other`.
		 * `other` becomes empty.
		 *************************************************************************************/

		vector_non_triv(vector_non_triv&& other) noexcept {
			size = other.size;
			capacity = other.capacity;
			data = (T*)malloc(sizeof(T) * capacity);
			for (size_t i = 0; i < size; ++i){
				new (data + i) T(std::move(other.data[i]));
				other.data[i].~T();
			}
			other.size = 0;
			other.capacity = 0;
			free(other.data);
			other.data = nullptr;
		}

		/*************************************************************************************
		 * COPY ASSIGNMENT: `operator=(const vector_non_triv& other)`
		 *
		 * Deep copy assignment from `other`.
		 * Frees current data, then copies size, capacity, and copy-constructs elements.
		 *
		 * Throws: std::bad_alloc on allocation failure, exceptions from T's copy constructor.
		 *************************************************************************************/

		vector_non_triv& operator=(const vector_non_triv& other) {
			if (data == other.data)
				return *this;
			size_t free_cap = size;
			size = other.size;
			capacity = other.capacity;
			try {
				for (size_t i = 0; i < free_cap; ++i)
					data[i].~T();
				free(data);
				data = (T*)malloc(sizeof(T) * capacity);
				for (size_t i = 0; i < other.size; ++i) {
					new (data + i) T(other.data[i]);
				}
			}

			catch (const std::bad_alloc& e) {
				if (data) {
					for (size_t i = 0; i < size; ++i)
						data[i].~T();
					free(data);
				}
				std::cerr << e.what() << std::endl;
				throw;
			}
			return *this;
		}

		/*************************************************************************************
		 * MOVE ASSIGNMENT: `operator=(vector_non_triv&& other) noexcept`
		 *
		 * Move assignment from `other`.
		 * Frees current data, then moves size, capacity, and move-constructs elements.
		 * `other` becomes empty.
		 *************************************************************************************/

		vector_non_triv& operator=(vector_non_triv&& other) noexcept {
			if (data == other.data)
				return *this;
			size_t free_cap = size;
			size = other.size;
			capacity = other.capacity;
			for (size_t i = 0; i < free_cap; ++i)
				data[i].~T();
			free(data);
			data = (T*)malloc(sizeof(T) * capacity);
			for (size_t i = 0; i < other.size; ++i) {
				new (data + i) T(std::move(other.data[i]));
				other.data[i].~T();
			}

			free(other.data);
			other.size = 0;
			other.capacity = 0;
			other.data = nullptr;
			return *this;
		}

		/*********************************************************************
		 * PUSH_BACK FUNCTIONS: `push_back(const T& elem)`, `push_back(T&& elem)`
		 *
		 * Appends element `elem` to the end.
		 * Reallocates memory if needed (using `normalize_capacity`).
		 * Overloads for copy and move semantics.
		 *
		 * Throws: std::bad_alloc on allocation failure, exceptions from T's constructors.
		 *********************************************************************/

		void push_back(const T& elem) override {
			if (size >= capacity) {
				data = normalize_capacity();
			}
			new (data + size) T(elem);
			++size;
		};
		void push_back(T&& elem) override {
			if (size >= capacity) {
				data = normalize_capacity();
			}
			new (data + size) T(std::move(elem));
			++size;
		};

		/*********************************************************************
		 * INDEXING FUNCTIONS: `at()`, `operator[]` (const and non-const)
		 *
		 * Access elements by index.
		 * `at()` - with bounds checking, throws `std::out_of_range`.
		 * `operator[]` - no bounds checking.
		 *********************************************************************/

		T& at(const size_t index) override {
			if (index >= size) {
				throw std::out_of_range("Index out of range");
			}
			return data[index];
		}
		const T& at(const size_t index) const override {
			if (index >= size) {
				throw std::out_of_range("Index out of range");
			}
			return data[index];
		};

		T& operator [] (const size_t index) override {
			return data[index];
		};
		const T& operator [] (const size_t index) const override {
			return data[index];
		};

		/*********************************************************************
		 * RESIZE FUNCTION: `resize(size_t new_size)`
		 *
		 * Changes vector size to `new_size`.
		 * Reallocates if needed, default-constructs new elements.
		 *
		 * Throws: std::bad_alloc on allocation failure, exceptions from T's constructor.
		 *********************************************************************/

		void resize(size_t new_size) override {
			if (new_size < size) {
				for (size_t i = new_size; i < size; ++i)
					data[i].~T();
				size = new_size;
			}
			else {
				while (new_size >= capacity) capacity *= 2;

				T* new_data = (T*)malloc(sizeof(T) * capacity);
				if (!new_data) throw std::bad_alloc();

				for (size_t i = 0; i < size; ++i) {
					new (new_data + i) T(std::move(data[i]));
					data[i].~T();
				}

				for(size_t i = size; i < new_size; ++i)
					new (new_data + i) T();
				free(data);
				size = new_size;
				data = new_data;
			}
		}

		/*********************************************************************
		 * RESERVE FUNCTION: `reserve(size_t reserve_size)`
		 *
		 * Reserves memory for at least `reserve_size` elements.
		 * Does not change `size`.
		 * Default-constructs elements in reserved space.
		 *
		 * Throws: std::bad_alloc on allocation failure, exceptions from T's constructor.
		 *********************************************************************/

		void reserve(size_t reserve_size) override {
			if (reserve_size < capacity)
				return;
			try {
				data = normalize_capacity(reserve_size);
			}
			catch (const std::bad_alloc& e) { std::cerr << e.what() << std::endl; throw; }
		}

		/*********************************************************************
		 * CLEAR FUNCTION: `clear()`
		 *
		 * Removes all elements, destroys them, and frees memory.
		 * Resets to initial state (capacity 1, one default-constructed element).
		 *
		 * Throws: std::bad_alloc on allocation failure, exceptions from T's constructor/destructor.
		 *********************************************************************/

		void clear() override {
			for (size_t i = 0; i < size; ++i)
				data[i].~T();
			free(data);
			size = 0;
			capacity = 1;
			data = (T*)malloc(sizeof(T));
		}

		/**************************************************************************************
		 * SHRINK_TO_FIT FUNCTION: `shrink_to_fit()`
		 *
		 * Reduces capacity to match `size`, minimizing memory.
		 * Destroys elements beyond `size`.
		 * Can be inefficient if called frequently.
		 *
		 * Throws: std::bad_alloc on allocation failure, exceptions from T's destructor.
		 **************************************************************************************/

		void shrink_to_fit() override {
			if (size == capacity || size == 0)
				return;
			void* raw = malloc(sizeof(T) * size);
			if (!raw) throw std::bad_alloc();

			T* new_data = static_cast<T*>(raw);

			for (size_t i = 0; i < size; ++i) {
				new (new_data + i) T(std::move(data[i]));
				data[i].~T();
			}

			free(data);
			data = new_data;
			capacity = size;
		}

		/*********************************************************************
		 * POP_BACK FUNCTION: `pop_back()`
		 *
		 * Removes the last element (decrements `size`), destroying it if needed.
		 * Frees memory and resets if the vector becomes empty.
		 *
		 * Throws: std::out_of_range if vector is empty, exceptions from T's destructor.
		 *********************************************************************/

		void pop_back() override {
			if (size == 0)
				throw std::out_of_range("Index out of range");
			--size;
			data[size].~T();
		}

		/*********************************************************************
		 * INSERT FUNCTIONS: `insert()` overloads
		 *
		 * Inserts element at given index/iterator.
		 * Shifts existing elements using move semantics.
		 * Reallocates if needed.
		 * Overloads for copy and move semantics, and index/iterator position.
		 *
		 * Throws: std::out_of_range, std::bad_alloc, exceptions from T's constructors.
		 *********************************************************************/

		void insert(size_t index, const T& value) override {
			if (index > size) {
				throw std::out_of_range("Index out of range");
			}
			if (index == size){
				push_back(value);
				return;
			}
			if (size == capacity) {
				data = normalize_capacity();
			}
			new (data + size) T(std::move(data[size - 1]));
			for (size_t i = size - 1; i > index; --i) {
				data[i] = std::move(data[i - 1]);
			}
			data[index] = value;
			++size;
		}
		void insert(size_t index, T&& value) override {
			if (index > size) {
				throw std::out_of_range("Index out of range");
			}
			if (index == size) {
				push_back(std::move(value));
				return;
			}
				
			if (size == capacity) {
				data = normalize_capacity();
			}
			new (data + size) T(std::move(data[size - 1]));
			for (size_t i = size - 1; i > index; --i) {
				data[i] = std::move(data[i - 1]);
			}
			data[index] = std::move(value);
			++size;
		}
		Iterator insert(Iterator pos, const T& value) override {
			size_t insert_index = pos - Iterator(data);
			if (insert_index > size) {
				throw std::out_of_range("Index out of range");
			}
			if (insert_index == size) {
				push_back(value);
				return Iterator(data + insert_index);
			}
				
			if (size == capacity) {
				data = normalize_capacity();
			}
			new (data + size) T(std::move(data[size - 1]));
			for (size_t i = size - 1; i > insert_index; --i) {
				data[i] = std::move(data[i - 1]);
			}
			data[insert_index] = value;
			++size;
			return Iterator(data + insert_index);
		}
		Iterator insert(Iterator pos, T&& value) override {
			size_t insert_index = pos - Iterator(data);
			if (insert_index > size) {
				throw std::out_of_range("Index out of range");
			}
			if (insert_index == size){
				push_back(std::move(value));
				return Iterator(data + insert_index);
			}
			if (size == capacity) {
				data = normalize_capacity();
			}
			new (data + size) T(std::move(data[size - 1]));
			for (size_t i = size - 1; i > insert_index; --i) {
				data[i] = std::move(data[i - 1]);
			}
			data[insert_index] = std::move(value);
			++size;
			return Iterator(data + insert_index);
		}

		/*********************************************************************
		 * ERASE FUNCTIONS: `erase()` overloads
		 *
		 * Erases element at given index/iterator, destroying it.
		 * Shifts subsequent elements using move semantics.
		 * Overloads for index and iterator position.
		 *
		 * Throws: std::out_of_range, exceptions from T's destructor.
		 *********************************************************************/

		void erase(size_t index) override {
			if (index >= size) {
				throw std::out_of_range("Index out of range");
			}
			data[index].~T();
			for (size_t i = index; i < size - 1; ++i) {
				data[i] = std::move(data[i + 1]);
			}
			--size;
			data[size].~T();
		}

		Iterator erase(Iterator pos) override {
			size_t erase_index = pos - Iterator(data);

			if (erase_index >= size) {
				throw std::out_of_range("Index out of range");
			}
			data[erase_index].~T();
			for (size_t i = erase_index; i < size - 1; ++i) {
				data[i] = std::move(data[i + 1]);
			}
			--size;
			data[size].~T();
			return pos;
		}

		/************************************************************
		 * SWAP FUNCTION: `swap(vector_base<T>& other) noexcept`
		 *
		 * Swaps data, size, and capacity with `other`.
		 * Fast, no element operations.
		 *************************************************************/

		void swap(vector_base<T>& other) noexcept {
			std::swap(size, other.size);
			std::swap(capacity, other.capacity);
			std::swap(data, other.data);
		};

		/// CHECKS IF OBJECT IS EMPTY, IF SIZE VARIABLE ISN'T HANDELED GOOD ENOUGH, CAN CAUSE CRUSH
		bool empty() { return size == 0 ? true : false; }

		/*********************************************************************
		 * DESTRUCTOR: `~vector_non_triv()`
		 *
		 * Destroys all constructed elements and prints a message.
		 * Frees allocated memory.
		 *
		 * Throws: exceptions from T's destructor.
		 *********************************************************************/

		~vector_non_triv() override { for (size_t i = 0; i < size; ++i) data[i].~T(); std::cout << "Destryed objects in vector_non_triv, next comes memory freeing" << std::endl; }
	};
}