#pragma once

namespace raw {
	/*
	 * @brief Base class for the raw::vector container.
	 *
	 * This class provides a common interface for dynamic array-based containers.
	 * It manages a raw memory buffer, storing elements of type T.
	 *
	 * ## Storage Structure:
	 * - `T* data` - Pointer to the allocated memory block.
	 * - `size_t size` - Number of elements currently stored.
	 * - `size_t capacity` - Total allocated capacity.
	 *
	 * ## Key Features:
	 * - Abstract methods for capacity management (`normalize_capacity`).
	 * - Virtual methods for basic container operations (`push_back`, `resize`, `reserve`, etc.).
	 * - Provides iterators (`IteratorBase`, `ReverseIterator`) for element traversal.
	 * - Implements memory management in the destructor.
	 *
	 * This class is intended to be inherited by specialized vector implementations
	 * handling trivial and non-trivial types differently.
	 */
	template<typename T> class vector_base;



	/*
	 * @brief Specialized class for raw::vector container handling trivial types.
	 *
	 * This class is a specialization of the base `vector` class that does not
	 * implement any new methods itself. Instead, it inherits the data members
	 * and virtual methods from the base class, overriding them to provide
	 * functionality specific to handling trivial types in raw memory.
	 *
	 * ## Storage Structure:
	 * - `T* data` - Pointer to the dynamically allocated memory block.
	 * - `size_t size` - Number of elements currently stored in the vector.
	 * - `size_t capacity` - Total capacity of the allocated memory block.
	 *
	 * ## Key Features:
	 * - Does not define new methods but overrides virtual methods from the base class.
	 * - Inherits raw memory management operations, such as resizing and memory reallocation.
	 * - Focuses on handling trivial types in raw memory, without additional overhead for complex types.
	 * - Provides basic vector functionalities like `push_back`, `resize`, and `reserve` by overriding base methods.
	 * - Implements iterators for element access and traversal.
	 *
	 * This class is intended to be a lightweight specialization for trivial types
	 * and serves as a base for further refinement in handling different type categories.
	 */
	template<typename T> class vector_triv;


	/*
	 * @brief Specialized class for raw::vector container handling non-trivial types.
	 *
	 * This class is a specialization of the base `vector` class that does not
	 * implement any new methods itself. Instead, it inherits the data members
	 * and virtual methods from the base class, overriding them to provide
	 * functionality specific to handling non-trivial types in raw memory.
	 *
	 * The class does not define new methods but simply overrides virtual methods
	 * from the base class to adapt them to the specific requirements of managing
	 * non-trivial types, which may involve calling their constructors and destructors.
	 *
	 * ## Storage Structure:
	 * - `T* data` - Pointer to the dynamically allocated memory block.
	 * - `size_t size` - Number of elements currently stored in the vector.
	 * - `size_t capacity` - Total capacity of the allocated memory block.
	 *
	 * ## Key Features:
	 * - Does not define new methods but overrides virtual methods from the base class.
	 * - Inherits raw memory management operations, such as resizing and memory reallocation.
	 * - Handles non-trivial types, ensuring proper construction, destruction, and potential memory management issues.
	 * - Provides basic vector functionalities like `push_back`, `resize`, and `reserve` by overriding base methods.
	 * - Implements iterators for element access and traversal.
	 *
	 * This class is designed for efficient handling of non-trivial types while
	 * leveraging base vector functionality, making it suitable for types with
	 * complex initialization or destruction behavior.
	 */
	template<typename T> class vector_non_triv;


	

}
