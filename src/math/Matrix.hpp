#pragma once

#include <cinttypes>
#include <execution>
#include <sstream>

#include <initializer_list>
#include "Vector.hpp"

namespace sw
{
    namespace math
    {

        template <typename M>
        class MatrixExpression
        {
        public:
            static constexpr bool is_leaf = false;

            auto operator[](size_t i) const
            {
                // Delegation to the actual expression type. This avoids dynamic polymorphism (a.k.a. virtual functions in C++)
                return static_cast<M const &>(*this)[i];
            }

            size_t size() const { return static_cast<M const &>(*this).size(); }

            template <typename U>
            auto transposeMult(const Vector<U> &rhs) const
            {
                // Take the linear combination of the columns of the transpose matrix.
                // Note that m_rows[i] is the i-th column of the transposed matrix.
                // Furthermore, to optimize the memory access the transposeMult is immediatly evaluated.
                Vector<U> sum = (*this)[0] * rhs[0];
                for (size_t i = 1; i < size(); ++i)
                {
                    sum += (*this)[i] * rhs[i];
                }
                return sum;
            }
        };

        template <typename T>
        class Matrix : public MatrixExpression<Matrix<T>>
        {
        public:
            static constexpr bool is_leaf = true;
            using type = typename ExtractMostInnerType<T>::type;

            Matrix() = default;

            Matrix(std::initializer_list<std::initializer_list<T>> initializerList)
            {
                size_t m_nRows = initializerList.size();
                // m_nCols = static_cast<uint32_t>(initializerList.begin()->size());

                m_rows = Vector<Vector<T>>(m_nRows);

                size_t rowIndex = 0;
                for (const auto &row : initializerList)
                {
                    m_rows[rowIndex] = Vector<T>(row);
                    rowIndex++;
                }
            }

            Matrix(typename std::vector<Vector<T>>::size_type nRows, typename std::vector<T>::size_type nCols) : m_rows(nRows, Vector<T>(nCols)) {}

            Matrix(typename std::vector<Vector<T>>::size_type nRows, typename std::vector<T>::size_type nCols, T initialValue) : m_rows(nRows, Vector<T>(nCols, initialValue)) {}

            // A Matrix can be constructed from any MatrixExpression, forcing its evaluation.
            template <typename M>
            NOINLINE Matrix(MatrixExpression<M> const &expr) : m_rows(expr.size())
            {
                // for (size_t i = 0; i != expr.size(); ++i)
                // {
                //     m_rows[i] = expr[i];
                // }

                std::for_each(std::execution::seq, m_rows.begin(), m_rows.end(), [&](Vector<T> &thisRow)
                              { 
                            size_t rowIndex = &thisRow - &m_rows[0];
                            thisRow = expr[rowIndex]; });
            }

            Vector<T> &operator[](size_t i) { return m_rows[i]; }

            const Vector<T> &operator[](size_t i) const { return m_rows[i]; }

            size_t size() const { return m_rows.size(); }

            size_t getNumRows() const { return m_rows.size(); }
            size_t getNumCols() const { return m_rows[0].size(); }

            // Comparison operators == and !=
            bool operator==(const Matrix<T> &other) const { return (m_rows == other.m_rows); }
            bool operator!=(const Matrix<T> &other) const { return (m_rows != other.m_rows); }

            std::string toString() const
            {
                return m_rows.toString();
            }

            friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix)
            {
                std::string str = matrix.toString();
                os << str;
                return os;
            }

            // __declspec(noinline) Matrix<T> &operator+=(const Matrix<T> &other)
            template <typename M>
            NOINLINE Matrix<T> &operator+=(const MatrixExpression<M> &other)
            {
                if (size() != other.size())
                {
                    throw std::invalid_argument("Number of matrix rows do not match for += operation");
                }

                // for (size_t i = 0; i < size(); ++i)
                // {
                //     m_rows[i] += other[i];
                // }

                std::for_each(std::execution::seq, m_rows.begin(), m_rows.end(), [&](Vector<T> &thisRow)
                              { 
                            size_t rowIndex = &thisRow - &m_rows[0];
                            thisRow += other[rowIndex]; });

                return *this;
            }

            void fill(T value)
            {
                for (Vector<T> &row : m_rows)
                {
                    row.fill(value);
                }
            }

            /**
             * @brief Create a random matrix with values uniformly distributed between min and max (inclusive).
             */
            static Matrix<T> rand(size_t nRows, size_t nCols, T min, T max)
            {
                Matrix<T> randomMatrix(nRows, nCols);
                for (uint32_t r = 0; r < nRows; ++r)
                {
                    randomMatrix[r] = Vector<T>::rand(nCols, min, max);
                }
                return randomMatrix;
            }

        private:
            Vector<Vector<T>> m_rows;
        };

        //-----------------------------------------------------------------------------------------
        // Matrix-Vector-Multiplication
        //-----------------------------------------------------------------------------------------
        template <typename M, typename V>
        class MatrixVectorProduct : public VectorExpression<MatrixVectorProduct<M, V>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<M::is_leaf, const M &, const M> m_m;
            std::conditional_t<V::is_leaf, const V &, const V> m_v;

        public:
            static constexpr bool is_leaf = false;
            using type = typename std::common_type<typename M::type, typename V::type>::type;

            MatrixVectorProduct(const M &m, const V &v) : m_m(m), m_v(v) {}

            auto operator[](size_t i) const
            {
                return m_m[i].dot(m_v);
            }

            size_t size() const { return m_m.size(); }
        };

        template <typename M, typename V>
        NOINLINE MatrixVectorProduct<M, V> operator*(const MatrixExpression<M> &m, const VectorExpression<V> &v)
        {
            return MatrixVectorProduct<M, V>(*static_cast<const M *>(&m), *static_cast<const V *>(&v));
        }

        //-----------------------------------------------------------------------------------------
        // Matrix-Scalar-Addition
        //-----------------------------------------------------------------------------------------
        template <typename M, arithmetic S>
        class MatrixScalarAddition : public MatrixExpression<MatrixScalarAddition<M, S>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<M::is_leaf, const M &, const M> m_m;
            const S m_s;

        public:
            static constexpr bool is_leaf = false;

            MatrixScalarAddition(const M &m, const S s) : m_m(m), m_s(s) {}

            auto operator[](size_t i) const
            {
                return m_m[i] + m_s;
            }

            size_t size() const { return m_m.size(); }
        };

        template <typename M, typename S>
        NOINLINE MatrixScalarAddition<M, S> operator+(const MatrixExpression<M> &m, const S s)
        {
            return MatrixScalarAddition<M, S>(*static_cast<const M *>(&m), s);
        }

        //-----------------------------------------------------------------------------------------
        // Matrix-Scalar-Product
        //-----------------------------------------------------------------------------------------
        template <typename M, arithmetic S>
        class MatrixScalarProduct : public MatrixExpression<MatrixScalarProduct<M, S>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<M::is_leaf, const M &, const M> m_m;
            const S m_s;

        public:
            static constexpr bool is_leaf = false;

            MatrixScalarProduct(const M &m, const S s) : m_m(m), m_s(s) {}

            auto operator[](size_t i) const
            {
                return m_m[i] * m_s;
            }

            size_t size() const { return m_m.size(); }
        };

        template <typename M, typename S>
        NOINLINE MatrixScalarProduct<M, S> operator*(const MatrixExpression<M> &m, const S s)
        {
            return MatrixScalarProduct<M, S>(*static_cast<const M *>(&m), s);
        }

        //-----------------------------------------------------------------------------------------
        // Vector-Vector Outer Product
        //-----------------------------------------------------------------------------------------
        template <typename U, typename V>
        class OuterProduct : public MatrixExpression<OuterProduct<U, V>>
        {
            // cref if leaf, copy otherwise
            std::conditional_t<U::is_leaf, const U &, const U> m_u;
            std::conditional_t<V::is_leaf, const V &, const V> m_v;

        public:
            static constexpr bool is_leaf = false;

            OuterProduct(const U &u, const V &v) : m_u(u), m_v(v) {}

            // Return row-i of the outer product
            auto operator[](size_t i) const
            {
                return m_v * m_u[i]; // changed order because we have the vector * scalar but not scalar * vector
            }

            size_t size() const { return m_u.size(); }
        };

        template <typename U, typename V>
        NOINLINE OuterProduct<U, V> outer(const VectorExpression<U> &u, const VectorExpression<V> &v)
        {
            return OuterProduct<U, V>(*static_cast<const U *>(&u), *static_cast<const V *>(&v));
        }

    } // namespace math
} // namespace sw
