#pragma once

#include <cinttypes>
#include <execution>
#include <sstream>

#include <initializer_list>
#include "Vector.hpp"

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
    };

    template <typename T>
    class Matrix : public MatrixExpression<Matrix<T>>
    {
    public:
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

        Vector<T> &operator[](size_t i) { return m_rows[i]; }

        const Vector<T> &operator[](size_t i) const { return m_rows[i]; }

        size_t size() const { return m_rows.size(); }

        friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix)
        {
            std::string str = matrix.m_rows.toString();
            os << str;
            return os;
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

        MatrixVectorProduct(const M &m, const V &v) : m_m(m), m_v(v) {}

        auto operator[](size_t i) const
        {
            return m_m[i].dot(m_v);
        }

        size_t size() const { return m_m.size(); }
    };

    template <typename M, typename V>
    MatrixVectorProduct<M, V>
    operator*(const MatrixExpression<M> &m, const VectorExpression<V> &v)
    {
        return MatrixVectorProduct<M, V>(*static_cast<const M *>(&m), *static_cast<const V *>(&v));
    }

} // namespace math