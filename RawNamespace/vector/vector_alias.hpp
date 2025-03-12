#pragma once
#include <type_traits>
#include "RawNamespace\RawBase.hpp"
#include "RawNamespace\vector\RawVector.hpp"
#include "RawNamespace\vector\trivial\RawVectorTriv.hpp"
#include "RawNamespace\vector\non-trvivial\RawVectorNonTriv.hpp"

namespace raw {
	/*********************************************************************
	 * vector_base ALIAS TEMPLATE: `vector_base<T>`
	 *
	 * Alias template for `vector_base` specialized on `T`.
	 * Chooses between `vector_triv` and `vector_non_triv` based on `myis_trivial`.
	 *********************************************************************/
	template <typename T>
	using vector = std::conditional_t<myis_trivial<T>::value, vector_triv<T>, vector_non_triv<T>>;
}