#pragma once

// #include <execution>
// #include <initializer_list>
// #include <random>
// #include <string>
// #include <type_traits>
#include <vector>

namespace math
{
    template <class T>
    concept arithmetic = std::is_arithmetic_v<T>;

    template <typename E>
    class VectorExpression
    {
    public:
        static constexpr bool is_leaf = false;

        double operator[](size_t i) const
        {
            // Delegation to the actual expression type. This avoids dynamic polymorphism (a.k.a. virtual functions in C++)
            int dummy = 3;
            return static_cast<E const &>(*this)[i];
        }
        size_t size() const { return static_cast<E const &>(*this).size(); }
    };

    /**
     * @brief Class to put a vector view on an std::vector.
     *
     * Let \f$v\f$ be a vector.
     *
     */
    template <typename T>
    class Vector : public VectorExpression<Vector<T>>
    {
    public:
        Vector(std::vector<T> &stdVector) : m_data(stdVector) {}

        T &operator[](size_t i) { return m_data[i]; }

        const T &operator[](size_t i) const { return m_data[i]; }

    private:
        std::vector<T> m_data;
    };
} // namespace math