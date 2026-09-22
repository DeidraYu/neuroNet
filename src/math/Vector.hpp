#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <vector>

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

        /**
         * @brief How many independent running totals an inner product loop keeps.
         *
         * A sum accumulated into one variable makes every step wait for the one before it,
         * and because floating point addition is not associative the compiler is not free
         * to reorder it, so the loop stays scalar. Keeping several totals that do not
         * depend on each other lets them go in the lanes of a vector register.
         *
         * Measured over one epoch of the first layer, with the instruction set of the build
         * machine enabled: one total 961 ms, four totals 184 ms, eight totals 84 ms.
         */
        constexpr size_t kAccumulators = 8;

        /**
         * @brief A dense vector over a contiguous block of memory.
         *
         * Operations evaluate immediately and return a new Vector. An earlier version built
         * an expression tree first and evaluated it on assignment, which avoided the
         * intermediate results but does not carry over to a flat, one-kernel-per-operation
         * layout. Measurement on this network put the difference in the noise, because
         * nearly all of the arithmetic sits in the matrix-vector product rather than in the
         * element-wise work around it. That version is kept for reference under
         * doc/expression-templates.
         */
        template <typename T>
        class Vector
        {
        public:
            using type = T;

            Vector() = default;

            explicit Vector(size_t sz) : m_data(sz) {}

            Vector(size_t sz, T initialValue) : m_data(sz, initialValue) {}

            Vector(const std::vector<T> &stdVector) : m_data(stdVector) {}

            Vector(std::initializer_list<T> initializerList) : m_data{initializerList} {}

            /**
             * @brief Convert element-wise from a vector of another arithmetic type.
             *
             * Explicit, so that a change of element type only ever happens where it is
             * written down. Operations do not mix types, so this constructor is the only
             * way across, and `Vector<double>(v) * 0.6` says plainly what it does.
             */
            template <typename U, typename = std::enable_if_t<!std::is_same_v<U, T>>>
            explicit Vector(const Vector<U> &other) : m_data(other.size())
            {
                for (size_t i = 0; i < m_data.size(); ++i)
                {
                    m_data[i] = static_cast<T>(other[i]);
                }
            }

            T &operator[](size_t i) { return m_data[i]; }
            const T &operator[](size_t i) const { return m_data[i]; }

            size_t size() const { return m_data.size(); }

            void resize(size_t sz) { m_data.resize(sz); }

            void fill(T value) { std::fill(m_data.begin(), m_data.end(), value); }

            bool operator==(const Vector<T> &other) const { return m_data == other.m_data; }
            bool operator!=(const Vector<T> &other) const { return m_data != other.m_data; }

            // Iterators
            typename std::vector<T>::iterator begin() { return m_data.begin(); }
            typename std::vector<T>::iterator end() { return m_data.end(); }
            typename std::vector<T>::const_iterator begin() const { return m_data.begin(); }
            typename std::vector<T>::const_iterator end() const { return m_data.end(); }
            auto cbegin() const { return m_data.cbegin(); }
            auto cend() const { return m_data.cend(); }

            const T *data() const { return m_data.data(); }
            T *data() { return m_data.data(); }

            //-------------------------------------------------------------------------------
            // Compound assignment
            //-------------------------------------------------------------------------------
            Vector<T> &operator+=(const Vector<T> &other)
            {
                if (size() != other.size())
                {
                    throw std::invalid_argument("Vector sizes do not match for += operation");
                }
                for (size_t i = 0; i < size(); ++i)
                {
                    m_data[i] += other[i];
                }
                return *this;
            }

            Vector<T> &operator-=(const Vector<T> &other)
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

            Vector<T> &operator*=(const Vector<T> &other)
            {
                if (size() != other.size())
                {
                    throw std::invalid_argument("Vector sizes do not match for *= operation");
                }
                for (size_t i = 0; i < size(); ++i)
                {
                    m_data[i] *= other[i];
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

            Vector<T> &operator-=(const T other)
            {
                for (size_t i = 0; i < size(); ++i)
                {
                    m_data[i] -= other;
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

            //-------------------------------------------------------------------------------
            // Reductions
            //-------------------------------------------------------------------------------

            /** @brief The inner product with another vector. */
            T dot(const Vector<T> &other) const
            {
                if (size() != other.size())
                {
                    throw std::invalid_argument("Vector sizes do not match for dot product");
                }

                // Several running totals rather than one. See the note on kAccumulators.
                T sums[kAccumulators] = {};

                const T *left = m_data.data();
                const T *right = other.data();
                const size_t n = size();

                size_t i = 0;
                for (; i + kAccumulators <= n; i += kAccumulators)
                {
                    for (size_t k = 0; k < kAccumulators; ++k)
                    {
                        sums[k] += left[i + k] * right[i + k];
                    }
                }

                T sum = T{0};
                for (size_t k = 0; k < kAccumulators; ++k)
                    sum += sums[k];

                for (; i < n; ++i)
                {
                    sum += left[i] * right[i];
                }
                return sum;
            }

            /** @brief The index of the largest element. */
            uint64_t argmax() const
            {
                const auto maxElement = std::max_element(m_data.begin(), m_data.end());
                return static_cast<uint64_t>(std::distance(m_data.begin(), maxElement));
            }

            //-------------------------------------------------------------------------------
            // Presentation and serialization
            //-------------------------------------------------------------------------------
            std::string toString() const
            {
                std::string str = "[";
                for (size_t i = 0; i < m_data.size(); ++i)
                {
                    str += std::to_string(m_data[i]);
                    if (i + 1 < m_data.size())
                        str += ", ";
                }
                str += "]";
                return str;
            }

            friend std::ostream &operator<<(std::ostream &os, const Vector<T> &vec)
            {
                os << vec.toString();
                return os;
            }

            TypeInfo getTypeInfo() const
            {
                return {std::type_index(typeid(T)), sizeof(T)};
            }

            std::vector<uint8_t> getBinaryString()
            {
                std::vector<uint8_t> binaryData;

                // Add the type information to binary data
                TypeInfo typeInfo = getTypeInfo();
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

            //-------------------------------------------------------------------------------
            // Construction
            //-------------------------------------------------------------------------------

            /**
             * @brief A vector with values uniformly distributed between min and max.
             *
             * The conversion from a random integer to a float is done here rather than by
             * std::uniform_real_distribution, which the standard leaves implementation
             * defined, so that a given seed produces the same values on every platform.
             */
            static Vector<T> rand(size_t sz, T min, T max)
            {
                Vector<T> randVector(sz);

                if constexpr (std::is_integral_v<T>)
                {
                    const uint64_t span = static_cast<uint64_t>(max) - static_cast<uint64_t>(min) + 1;
                    for (size_t i = 0; i < sz; ++i)
                    {
                        randVector[i] = static_cast<T>(static_cast<uint64_t>(min) +
                                                       Random::below(static_cast<uint32_t>(span)));
                    }
                }
                else
                {
                    for (size_t i = 0; i < sz; ++i)
                    {
                        const uint32_t bits = static_cast<uint32_t>(Random::gen());
                        const double u = bits / 4294967296.0; // [0, 1), exact: 2^32 is a power of two
                        randVector[i] = static_cast<T>(min + u * (max - min));
                    }
                }
                return randVector;
            }

        private:
            std::vector<T> m_data;
        };

        //-----------------------------------------------------------------------------------------
        // Vector and vector
        //
        // Both sides must have the same element type. Mixing them would quietly widen a
        // whole vector, so a change of type has to be written out with the converting
        // constructor instead: Vector<double>(v) * 0.6
        //-----------------------------------------------------------------------------------------
        template <typename T>
        Vector<T> operator+(const Vector<T> &u, const Vector<T> &v)
        {
            assert(u.size() == v.size());
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = u[i] + v[i];
            return result;
        }

        template <typename T>
        Vector<T> operator-(const Vector<T> &u, const Vector<T> &v)
        {
            assert(u.size() == v.size());
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = u[i] - v[i];
            return result;
        }

        template <typename T>
        Vector<T> operator*(const Vector<T> &u, const Vector<T> &v)
        {
            assert(u.size() == v.size());
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = u[i] * v[i];
            return result;
        }

        //-----------------------------------------------------------------------------------------
        // Vector and scalar
        //
        // The scalar has the element type of the vector, for the same reason.
        //-----------------------------------------------------------------------------------------
        template <typename T>
        Vector<T> operator+(const Vector<T> &u, const T s)
        {
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = u[i] + s;
            return result;
        }

        template <typename T>
        Vector<T> operator-(const Vector<T> &u, const T s)
        {
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = u[i] - s;
            return result;
        }

        template <typename T>
        Vector<T> operator*(const Vector<T> &u, const T s)
        {
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = u[i] * s;
            return result;
        }

        //-----------------------------------------------------------------------------------------
        // Element-wise functions
        //-----------------------------------------------------------------------------------------
        template <typename T>
        Vector<T> sigmoid(const Vector<T> &u)
        {
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = sigmoid(u[i]);
            return result;
        }

        template <typename T>
        Vector<T> sigmoid_prime(const Vector<T> &u)
        {
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = sigmoid_prime(u[i]);
            return result;
        }

        template <typename T>
        Vector<T> sin(const Vector<T> &u)
        {
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = static_cast<T>(std::sin(u[i]));
            return result;
        }

        template <typename T>
        Vector<T> cos(const Vector<T> &u)
        {
            Vector<T> result(u.size());
            for (size_t i = 0; i < u.size(); ++i)
                result[i] = static_cast<T>(std::cos(u[i]));
            return result;
        }

    } // namespace math
} // namespace sw
