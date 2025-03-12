#pragma once
#include <type_traits>

namespace raw {
	/************************************************************************
	* typename created to handle different types, trivial and non-trivial
	* sets a value based on the copyability and destructability
	***********************************************************************/
	template<typename T>
	struct myis_trivial {
		static const bool value = std::is_trivially_copyable<T>::value && std::is_trivially_destructible<T>::value;
	};
}