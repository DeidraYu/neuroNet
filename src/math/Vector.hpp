#pragma once

// #include <execution>
// #include <initializer_list>
// #include <type_traits>

#include <cassert>
#include <cmath>
#include <execution>
#include <functional>
#include <iostream>
// #include <random>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <vector>
#include <variant>

#include "utils.hpp"

namespace sw
{
    namespace math
    {
        struct TypeInfo
        {
            std::type_index index;
            size_t size;
        };

        // Forward declaration of Vector for the special case
        template <typename T>
        class Vector;

        // Generic implementation for the case when T is not a VectorExpression or VectorVectorAddition
        template <typename T>
        struct ExtractMostInnerType
        {
            using type = T;
        };

        // Specialization for the case when the template argument is VectorExpression<T>
        // Recursive metafunction to extract the most inner type from nested Vectors
        template <typename T>
        struct ExtractMostInnerType<Vector<T>>
        {
            using type = typename ExtractMostInnerType<T>::type;
        };

        template <typename V>
        class VectorExpression
        {
        public:
            static constexpr bool is_leaf = false;

            auto operator[](size_t i) const
            {
                // Delegation to the actual expression type. This avoids dynamic polymorphism (a.k.a. virtual functions in C++)
                return static_cast<V const &>(*this)[i];
            }

            // Iterators
            // typename std::vector<T>::iterator begin() { return static_cast<V const &>(*this).begin(); }
            // typename std::vector<T>::iterator end() { return static_cast<V const &>(*this).end(); }
            // auto begin() { return static_cast<V const &>(*this).begin(); }
            auto end() { return static_cast<V const &>(*this).end(); }

            // Constant iterators
            auto cbegin() const { return static_cast<V const &>(*this).cbegin(); }
            auto cend() const { return static_cast<V const &>(*this).cend(); }

            // Design choice: The dot product is not Lazy. Calling the .dot() operation enforces it's children to be immediatly
            //                being evaluated. This avoids repeated computation of a single scalar. It is possible that we need
            //                to come back on this decision at a later moment.
            template <typename V2>
            auto dot(const V2 &other) const
            {
                // Compute the inner product in parallel using C++20's execution policies
                using thisType = typename V::type;
                using otherType = typename V2::type;
                using resultType = typename std::common_type<thisType, otherType>::type;

                return std::transform_reduce(std::execution::seq,
                                             cbegin(), cend(),
                                             other.cbegin(),
                                             resultType{0}, // Initial value for the sum
                                             std::plus<>(), // Binary operation (sum)
                                             [](thisType x, otherType y)
                                             { return x * y; }); // Element-wise multiplication
            }

            size_t size() const { return static_cast<V const &>(*this).size(); }
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
            static constexpr bool is_leaf = true;

            using type = typename ExtractMostInnerType<T>::type;

            Vector() = default;

            Vector(typename std::vector<T>::size_type sz) : m_data(sz) {}

            Vector(typename std::vector<T>::size_type sz, T initialValue) : m_data(sz, initialValue) {}

            Vector(std::vector<T> &stdVector) : m_data(stdVector) {}

            Vector(std::initializer_list<T> initializerList) : m_data{initializerList} {}

            // A Vec can be constructed from any VecExpression, forcing its evaluation.
            template <typename V>
            Vector(VectorExpression<V> const &expr) : m_data(expr.size())
            {
                std::for_each(std::execution::seq, m_data.begin(), m_data.end(), [&](auto &thisElement)
                              { 
                            size_t index = &thisElement - &m_data[0];
                            thisElement = static_cast<T>(expr[index]); });
            }

            template <typename V1>
            Vector<T> &operator+=(const VectorExpression<V1> &other)
            {
                if (size() != other.size())
                {
                    throw std::invalid_argument("Vector sizes do not match for += operation");
                }

                for (size_t i = 0; i < size(); ++i)
                {
                    m_data[i] += static_cast<T>(other[i]);
                }

                return *this;
            }

            Vector<T> &operator+=(const T other)
            {
                for (size_t i = 0; i < size(); ++i)
                {
                    m_data[i] += other;
                }

                return *this;
            }

            template <typename V1>
            Vector<T> &operator-=(const VectorExpression<V1> &other)
            {
                if (size() != other.size())
                {
                    throw std::invalid_argument("Vector sizes do not match for -= operation");
                }

                for (size_t i = 0; i < size(); ++i)
                {
                    m_data[i] -= other[i];
                }

                return *this;
            }

            Vector<T> &operator-=(const T other)
            {
                for (size_t i = 0; i < size(); ++i)
                {
                    m_data[i] -= other;
                }

                return *this;
            }

            template <typename V1>
            Vector<T> &operator*=(const VectorExpression<V1> &other)
            {
                if (size() != other.size())
                {
                    throw std::invalid_argument("Vector sizes do not match for += operation");
                }

                for (size_t i = 0; i < size(); ++i)
                {
                    m_data[i] *= other[i];
                }

                return *this;
            }

            Vector<T> &operator*=(const T other)
            {
                for (size_t i = 0; i < size(); ++i)
                {
                    m_data[i] *= other;
                }

                return *this;
            }

            T &operator[](size_t i) { return m_data[i]; }

            const T &operator[](size_t i) const { return m_data[i]; }

            size_t size() const { return m_data.size(); }

            // Comparison operators == and !=
            bool operator==(const Vector<T> &other) const { return (m_data == other.m_data); }

            // Helper function to convert elements to string
            template <typename U>
            std::string elementToString(const U &value) const
            {
                return std::to_string(value);
            }

            template <typename U>
            std::string elementToString(const Vector<U> &vector) const
            {
                return vector.toString();
            }

            // Function to convert the Vector to a string representation
            std::string toString() const
            {
                std::string str = "[";
                for (size_t i = 0; i < m_data.size(); ++i)
                {
                    str += elementToString(m_data[i]);
                    if (i < m_data.size() - 1)
                        str += ", ";
                }
                str += "]";
                return str;
            }

            void fill(T value)
            {
                std::fill(m_data.begin(), m_data.end(), value);
            }
            // Iterators
            typename std::vector<T>::iterator begin() { return m_data.begin(); }
            typename std::vector<T>::iterator end() { return m_data.end(); }

            // Constant iterators
            auto cbegin() const { return m_data.cbegin(); }
            auto cend() const { return m_data.cend(); }

            // Overload the << operator for output
            friend std::ostream &operator<<(std::ostream &os, const Vector<T> &vec)
            {
                os << vec.toString();
                return os;
            }

            /**
             * @brief Create a random vector with values uniformly distributed between min and max (inclusive).
             */
            static Vector<T> rand(typename std::vector<T>::size_type sz, T min, T max)
            {
                Vector<T> randVector(sz);

                if constexpr (std::is_integral_v<T>)
                {
                    // For some reason the std does not define the std::uniform_int_distribution<> for the uint8_t type.
                    using CommonType = typename std::common_type<T, uint16_t>::type;
                    std::uniform_int_distribution<CommonType> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

                    for (uint32_t i = 0; i < sz; ++i)
                    {
                        randVector[i] = static_cast<T>(dis(Random::gen));
                    }
                }
                else if constexpr (std::is_floating_point_v<T>)
                {
                    // Note: std::uniform_real_distribution is deliberately left implementation-defined
                    // by the standard, so libstdc++ and the MSVC STL turn the same engine output into
                    // different floats. The engine itself (std::mt19937) is specified exactly, so we do
                    // the integer-to-float conversion here to keep results identical on every platform.
                    for (uint32_t i = 0; i < sz; ++i)
                    {
                        const uint32_t bits = static_cast<uint32_t>(Random::gen());
                        const double u = bits / 4294967296.0; // [0, 1), exact: 2^32 is a power of two
                        randVector[i] = static_cast<T>(min + u * (max - min));
                    }
                }
                return randVector;
            }

            uint64_t argmax()
            {
                auto maxElement = std::max_element(begin(), end());
                uint64_t index = std::distance(begin(), maxElement);
                return index;
            }

            const T *data() const
            {
                return m_data.data();
            }

            template <typename V>
            TypeInfo getTypeInfo()
            {
                return {std::type_index(typeid(T)), sizeof(T)};
            }

            std::vector<uint8_t> getBinaryString()
            {
                std::vector<uint8_t> binaryData;

                // Add the type information to binary data
                TypeInfo typeInfo = getTypeInfo<T>();
                const uint8_t *typeInfoPtr = reinterpret_cast<const uint8_t *>(&typeInfo);
                binaryData.insert(binaryData.end(), typeInfoPtr, typeInfoPtr + sizeof(TypeInfo));

                // Add the number of elements to binary data
                size_t numElements = m_data.size();
                const uint8_t *numElementsPtr = reinterpret_cast<const uint8_t *>(&numElements);
                binaryData.insert(binaryData.end(), numElementsPtr, numElementsPtr + sizeof(size_t));

                // Add the vector data to binary data
                const uint8_t *dataPtr = reinterpret_cast<const uint8_t *>(m_data.data());
                binaryData.insert(binaryData.end(), dataPtr, dataPtr + numElements * sizeof(T));

                return binaryData;
            }

        private:
            std::vector<T> m_data;
        };

        //-----------------------------------------------------------------------------------------
        // Vector-Vector-Addition
        //-----------------------------------------------------------------------------------------
        template <typename V1, typename V2>
        class VectorVectorAddition : public VectorExpression<VectorVectorAddition<V1, V2>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<V1::is_leaf, const V1 &, const V1> m_u;
            std::conditional_t<V2::is_leaf, const V2 &, const V2> m_v;

        public:
            static constexpr bool is_leaf = false;
            using type = typename std::common_type<typename V1::type, typename V2::type>::type;

            VectorVectorAddition(V1 const &u, V2 const &v) : m_u(u), m_v(v)
            {
                assert(u.size() == v.size());
            }

            auto operator[](size_t i) const
            {
                return m_u[i] + m_v[i];
            }

            size_t size() const { return m_v.size(); }
        };

        template <typename V1, typename V2>
        VectorVectorAddition<V1, V2>
        operator+(VectorExpression<V1> const &u, VectorExpression<V2> const &v)
        {
            return VectorVectorAddition<V1, V2>(*static_cast<const V1 *>(&u), *static_cast<const V2 *>(&v));
        }

        //-----------------------------------------------------------------------------------------
        // Vector-Scalar-Addition
        //-----------------------------------------------------------------------------------------

        template <typename V, arithmetic S>
        class VectorScalarAddition : public VectorExpression<VectorScalarAddition<V, S>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<V::is_leaf, const V &, const V> m_u;
            S m_s;

        public:
            static constexpr bool is_leaf = false;
            using type = typename std::common_type<typename V::type, S>::type;

            VectorScalarAddition(V const &u, const S s) : m_u(u), m_s(s){};

            auto operator[](size_t i) const { return m_u[i] + m_s; }

            size_t size() const { return m_u.size(); }
        };

        template <typename V, typename S>
        VectorScalarAddition<V, S>
        operator+(VectorExpression<V> const &u, const S s)
        {
            return VectorScalarAddition<V, S>(*static_cast<const V *>(&u), s);
        }

        //-----------------------------------------------------------------------------------------
        // Vector-Vector-Subtraction
        //-----------------------------------------------------------------------------------------
        template <typename V1, typename V2>
        class VectorVectorSubtraction : public VectorExpression<VectorVectorSubtraction<V1, V2>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<V1::is_leaf, const V1 &, const V1> m_u;
            std::conditional_t<V2::is_leaf, const V2 &, const V2> m_v;

        public:
            static constexpr bool is_leaf = false;
            using type = typename std::common_type<typename V1::type, typename V2::type>::type;

            VectorVectorSubtraction(V1 const &u, V2 const &v) : m_u(u), m_v(v)
            {
                assert(u.size() == v.size());
            }

            auto operator[](size_t i) const
            {
                return m_u[i] - m_v[i];
            }

            size_t size() const { return m_v.size(); }
        };

        template <typename V1, typename V2>
        VectorVectorSubtraction<V1, V2>
        operator-(VectorExpression<V1> const &u, VectorExpression<V2> const &v)
        {
            return VectorVectorSubtraction<V1, V2>(*static_cast<const V1 *>(&u), *static_cast<const V2 *>(&v));
        }

        //-----------------------------------------------------------------------------------------
        // Vector-Scalar-Subtraction
        //-----------------------------------------------------------------------------------------
        template <typename V, arithmetic S>
        class VectorScalarSubtraction : public VectorExpression<VectorScalarSubtraction<V, S>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<V::is_leaf, const V &, const V> m_u;
            S m_s;

        public:
            static constexpr bool is_leaf = false;
            using type = typename std::common_type<typename V::type, S>::type;

            VectorScalarSubtraction(V const &u, const S s) : m_u(u), m_s(s){};

            auto operator[](size_t i) const { return m_u[i] - m_s; }

            size_t size() const { return m_u.size(); }
        };

        template <typename V, typename S>
        VectorScalarSubtraction<V, S>
        operator-(VectorExpression<V> const &u, const S s)
        {
            return VectorScalarSubtraction<V, S>(*static_cast<const V *>(&u), s);
        }

        //-----------------------------------------------------------------------------------------
        // Vector-Vector-Multiplicatation
        //-----------------------------------------------------------------------------------------
        template <typename V1, typename V2>
        class VectorVectorMultiplication : public VectorExpression<VectorVectorMultiplication<V1, V2>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<V1::is_leaf, const V1 &, const V1> m_u;
            std::conditional_t<V2::is_leaf, const V2 &, const V2> m_v;

        public:
            static constexpr bool is_leaf = false;
            using type = typename std::common_type<typename V1::type, typename V2::type>::type;

            VectorVectorMultiplication(V1 const &u, V2 const &v) : m_u(u), m_v(v)
            {
                assert(u.size() == v.size());
            }

            auto operator[](size_t i) const { return m_u[i] * m_v[i]; }

            size_t size() const { return m_v.size(); }
        };

        template <typename V1, typename V2>
        VectorVectorMultiplication<V1, V2>
        operator*(VectorExpression<V1> const &u, VectorExpression<V2> const &v)
        {
            return VectorVectorMultiplication<V1, V2>(*static_cast<const V1 *>(&u), *static_cast<const V2 *>(&v));
        }

        //-----------------------------------------------------------------------------------------
        // Vector-Scalar-Multiplication
        //-----------------------------------------------------------------------------------------
        template <typename V, arithmetic S>
        class VectorScalarMultiplication : public VectorExpression<VectorScalarMultiplication<V, S>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<V::is_leaf, const V &, const V> m_u;
            S m_s;

        public:
            static constexpr bool is_leaf = false;
            using type = typename std::common_type<typename V::type, S>::type;

            VectorScalarMultiplication(V const &u, const S s) : m_u(u), m_s(s){};

            auto operator[](size_t i) const { return m_u[i] * m_s; }

            size_t size() const { return m_u.size(); }
        };

        template <typename V, typename S>
        VectorScalarMultiplication<V, S>
        operator*(VectorExpression<V> const &u, const S s)
        {
            return VectorScalarMultiplication<V, S>(*static_cast<const V *>(&u), s);
        }

        //-----------------------------------------------------------------------------------------
        // VectorFunction
        //-----------------------------------------------------------------------------------------
        template <typename U>
        class VectorFunction : public VectorExpression<VectorFunction<U>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<U::is_leaf, const U &, const U> m_u;

        public:
            static constexpr bool is_leaf = false;
            using type = typename U::type;

            VectorFunction(const U &u, std::function<type(type)> function) : m_u(u), m_function(function) {}

            auto operator[](size_t i) const
            {
                return m_function(m_u[i]);
            }

            size_t size() const { return m_u.size(); }

        private:
            std::function<type(type)> m_function;
        };

        template <typename U>
        VectorFunction<U> sigmoid(const VectorExpression<U> &u)
        {
            return VectorFunction<U>(*static_cast<const U *>(&u), [](typename U::type x) -> typename U::type
                                     { return sigmoid(x); });
        }

        template <typename U>
        VectorFunction<U> sigmoid_prime(const VectorExpression<U> &u)
        {
            return VectorFunction<U>(*static_cast<const U *>(&u), [](typename U::type x) -> typename U::type
                                     { return sigmoid_prime(x); });
        }

        template <typename U>
        VectorFunction<U> sin(const VectorExpression<U> &u)
        {
            return VectorFunction<U>(*static_cast<const U *>(&u), [](typename U::type x) -> typename U::type
                                     { return std::sin(x); });
        }

        template <typename U>
        VectorFunction<U> cos(const VectorExpression<U> &u)
        {
            return VectorFunction<U>(*static_cast<const U *>(&u), [](typename U::type x) -> typename U::type
                                     { return std::cos(x); });
        }

    } // namespace math
} // namespace sw
