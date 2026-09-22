#pragma once

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Vector.hpp"

namespace sw
{
    namespace math
    {
        /**
         * @brief A view on one row of a Matrix.
         *
         * The rows of a Matrix are not separate objects, so indexing a Matrix hands back a
         * window onto part of its buffer rather than a Vector. It is only a pointer and a
         * length, so it is cheap to pass around, but it does not own anything: it stops
         * being valid as soon as the Matrix it came from is resized or destroyed.
         */
        template <typename T>
        class RowView
        {
        public:
            RowView(T *data, size_t length) : m_data(data), m_length(length) {}

            T &operator[](size_t j) { return m_data[j]; }
            const T &operator[](size_t j) const { return m_data[j]; }

            size_t size() const { return m_length; }

            T *begin() { return m_data; }
            T *end() { return m_data + m_length; }
            const T *begin() const { return m_data; }
            const T *end() const { return m_data + m_length; }

            const T *data() const { return m_data; }
            T *data() { return m_data; }

            /** @brief Copy the contents of a Vector into this row. */
            RowView<T> &operator=(const Vector<T> &v)
            {
                assert(v.size() == m_length);
                for (size_t j = 0; j < m_length; ++j)
                    m_data[j] = v[j];
                return *this;
            }

            /** @brief A copy of this row as a standalone Vector. */
            Vector<T> toVector() const
            {
                Vector<T> v(m_length);
                for (size_t j = 0; j < m_length; ++j)
                    v[j] = m_data[j];
                return v;
            }

        private:
            T *m_data;
            size_t m_length;
        };

        /**
         * @brief The outer product of two vectors, recorded but not yet formed.
         *
         * Back propagation accumulates an outer product into the weight gradient once per
         * training sample. Forming the matrix first would allocate and fill one the size of
         * the whole weight matrix every time, which measured at roughly a third of the
         * training time. Keeping the two vectors and adding the product in directly costs
         * nothing and reads the same at the call site.
         *
         * This holds references, so it is meant to be used in the expression that creates
         * it. It is the one place where an operation is deferred rather than evaluated at
         * once, and it is here because it was measured, not on principle.
         */
        template <typename T>
        class OuterProduct
        {
        public:
            OuterProduct(const Vector<T> &u, const Vector<T> &v) : m_u(u), m_v(v) {}

            const Vector<T> &u() const { return m_u; }
            const Vector<T> &v() const { return m_v; }

            size_t getNumRows() const { return m_u.size(); }
            size_t getNumCols() const { return m_v.size(); }

        private:
            const Vector<T> &m_u;
            const Vector<T> &m_v;
        };

        /**
         * @brief A dense matrix held row by row in one contiguous block of memory.
         *
         * Element (i, j) lives at index i * numCols + j. Keeping the whole matrix in a
         * single allocation, rather than as a vector of row vectors, is what allows a row to
         * be walked contiguously, allows the inner loop of a product to be vectorized, and
         * allows the whole thing to be handed to a device as one buffer later on.
         */
        template <typename T>
        class Matrix
        {
        public:
            using type = T;

            Matrix() = default;

            Matrix(size_t nRows, size_t nCols)
                : m_rows(nRows), m_cols(nCols), m_data(nRows * nCols) {}

            Matrix(size_t nRows, size_t nCols, T initialValue)
                : m_rows(nRows), m_cols(nCols), m_data(nRows * nCols, initialValue) {}

            Matrix(std::initializer_list<std::initializer_list<T>> initializerList)
            {
                m_rows = initializerList.size();
                m_cols = m_rows == 0 ? 0 : initializerList.begin()->size();
                m_data.resize(m_rows * m_cols);

                size_t i = 0;
                for (const auto &row : initializerList)
                {
                    if (row.size() != m_cols)
                    {
                        throw std::invalid_argument("All rows of a Matrix must have the same length");
                    }
                    size_t j = 0;
                    for (const auto &value : row)
                    {
                        m_data[i * m_cols + j] = value;
                        ++j;
                    }
                    ++i;
                }
            }

            /** @brief Form the matrix of an outer product. */
            Matrix(const OuterProduct<T> &product)
                : m_rows(product.getNumRows()), m_cols(product.getNumCols()),
                  m_data(product.getNumRows() * product.getNumCols())
            {
                const T *vData = product.v().data();
                for (size_t i = 0; i < m_rows; ++i)
                {
                    const T factor = product.u()[i];
                    T *row = &m_data[i * m_cols];
                    for (size_t j = 0; j < m_cols; ++j)
                    {
                        row[j] = factor * vData[j];
                    }
                }
            }

            //-------------------------------------------------------------------------------
            // Access
            //-------------------------------------------------------------------------------
            T &operator()(size_t i, size_t j) { return m_data[i * m_cols + j]; }
            const T &operator()(size_t i, size_t j) const { return m_data[i * m_cols + j]; }

            RowView<T> operator[](size_t i) { return RowView<T>(&m_data[i * m_cols], m_cols); }
            RowView<const T> operator[](size_t i) const { return RowView<const T>(&m_data[i * m_cols], m_cols); }

            size_t getNumRows() const { return m_rows; }
            size_t getNumCols() const { return m_cols; }

            /** @brief The number of rows, so that a Matrix can be indexed like a vector of rows. */
            size_t size() const { return m_rows; }

            const T *data() const { return m_data.data(); }
            T *data() { return m_data.data(); }

            void fill(T value) { std::fill(m_data.begin(), m_data.end(), value); }

            bool operator==(const Matrix<T> &other) const
            {
                return m_rows == other.m_rows && m_cols == other.m_cols && m_data == other.m_data;
            }
            bool operator!=(const Matrix<T> &other) const { return !(*this == other); }

            //-------------------------------------------------------------------------------
            // Compound assignment
            //-------------------------------------------------------------------------------
            Matrix<T> &operator+=(const Matrix<T> &other)
            {
                if (m_rows != other.m_rows || m_cols != other.m_cols)
                {
                    throw std::invalid_argument("Matrix shapes do not match for += operation");
                }
                for (size_t k = 0; k < m_data.size(); ++k)
                {
                    m_data[k] += other.m_data[k];
                }
                return *this;
            }

            /** @brief Add an outer product without forming it as a matrix first. */
            Matrix<T> &operator+=(const OuterProduct<T> &product)
            {
                if (m_rows != product.getNumRows() || m_cols != product.getNumCols())
                {
                    throw std::invalid_argument("Outer product shape does not match for += operation");
                }

                // A plain loop, unlike the inner product. Each element is written once and
                // nothing is accumulated across steps, so a compiler vectorizes this without
                // being told the steps are independent.
                const T *vData = product.v().data();
                for (size_t i = 0; i < m_rows; ++i)
                {
                    const T factor = product.u()[i];
                    T *row = &m_data[i * m_cols];
                    for (size_t j = 0; j < m_cols; ++j)
                    {
                        row[j] += factor * vData[j];
                    }
                }
                return *this;
            }

            //-------------------------------------------------------------------------------
            // Products
            //-------------------------------------------------------------------------------

            /**
             * @brief The product of the transpose of this matrix with a vector.
             *
             * This is the backward direction of the layer, where the gradient is carried from
             * the outputs to the inputs. Rather than transposing the matrix, which would copy
             * it, the result is accumulated column by column so that the inner loop still
             * walks a row of the stored matrix contiguously.
             */
            Vector<T> transposeMult(const Vector<T> &v) const
            {
                if (v.size() != m_rows)
                {
                    throw std::invalid_argument("Vector length does not match for transposeMult");
                }

                Vector<T> result(m_cols, T{0});
                T *out = result.data();
                for (size_t i = 0; i < m_rows; ++i)
                {
                    const T *row = &m_data[i * m_cols];
                    const T factor = v[i];
                    for (size_t j = 0; j < m_cols; ++j)
                    {
                        out[j] += row[j] * factor;
                    }
                }
                return result;
            }

            //-------------------------------------------------------------------------------
            // Presentation
            //-------------------------------------------------------------------------------
            std::string toString() const
            {
                std::string str = "[";
                for (size_t i = 0; i < m_rows; ++i)
                {
                    str += "[";
                    for (size_t j = 0; j < m_cols; ++j)
                    {
                        str += std::to_string(m_data[i * m_cols + j]);
                        if (j + 1 < m_cols)
                            str += ", ";
                    }
                    str += "]";
                    if (i + 1 < m_rows)
                        str += ", ";
                }
                str += "]";
                return str;
            }

            friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix)
            {
                os << matrix.toString();
                return os;
            }

            //-------------------------------------------------------------------------------
            // Construction
            //-------------------------------------------------------------------------------

            /** @brief A matrix with values uniformly distributed between min and max. */
            static Matrix<T> rand(size_t nRows, size_t nCols, T min, T max)
            {
                Matrix<T> randomMatrix(nRows, nCols);
                const Vector<T> values = Vector<T>::rand(nRows * nCols, min, max);
                for (size_t k = 0; k < values.size(); ++k)
                {
                    randomMatrix.m_data[k] = values[k];
                }
                return randomMatrix;
            }

        private:
            size_t m_rows{0};
            size_t m_cols{0};
            std::vector<T> m_data;
        };

        //-----------------------------------------------------------------------------------------
        // Matrix and vector
        //
        // As with vectors, both sides carry the same element type. A change of type is
        // written out with the converting constructor rather than happening inside an
        // operation.
        //-----------------------------------------------------------------------------------------

        /**
         * @brief Matrix times vector.
         *
         * One output per row, each the inner product of that row with the vector. The inner
         * loop walks the row contiguously, which is what lets the compiler vectorize it.
         */
        template <typename T>
        Vector<T> operator*(const Matrix<T> &A, const Vector<T> &x)
        {
            if (A.getNumCols() != x.size())
            {
                throw std::invalid_argument("Matrix and vector shapes do not match for product");
            }

            const size_t nCols = A.getNumCols();
            const T *xData = x.data();

            Vector<T> result(A.getNumRows());
            for (size_t i = 0; i < A.getNumRows(); ++i)
            {
                const T *row = A.data() + i * nCols;

                // Several running totals rather than one. Accumulating into a single total
                // makes every step wait for the one before it, and since floating point
                // addition is not associative the compiler is not allowed to reorder that,
                // so the loop stays scalar however wide the registers are. Totals that do
                // not depend on each other can go in separate lanes.
                //
                // Measured over one epoch of these products: one total 961 ms, four 184 ms,
                // eight 84 ms. The count is kAccumulators, in Vector.hpp.
                //
                // This wants the build to target the instruction set of the machine, which
                // is what NEURONET_TUNE_FOR_THIS_MACHINE does in CMakeLists.txt. Without it
                // the vector registers are only 128 bits wide and much of the gain is lost.
                T sums[kAccumulators] = {};

                size_t j = 0;
                for (; j + kAccumulators <= nCols; j += kAccumulators)
                {
                    for (size_t k = 0; k < kAccumulators; ++k)
                    {
                        sums[k] += row[j + k] * xData[j + k];
                    }
                }

                T sum = T{0};
                for (size_t k = 0; k < kAccumulators; ++k)
                    sum += sums[k];

                for (; j < nCols; ++j)
                {
                    sum += row[j] * xData[j];
                }
                result[i] = sum;
            }
            return result;
        }

        //-----------------------------------------------------------------------------------------
        // Matrix and scalar
        //-----------------------------------------------------------------------------------------
        template <typename T>
        Matrix<T> operator+(const Matrix<T> &A, const T s)
        {
            Matrix<T> result(A.getNumRows(), A.getNumCols());
            for (size_t i = 0; i < A.getNumRows(); ++i)
                for (size_t j = 0; j < A.getNumCols(); ++j)
                    result(i, j) = A(i, j) + s;
            return result;
        }

        template <typename T>
        Matrix<T> operator*(const Matrix<T> &A, const T s)
        {
            Matrix<T> result(A.getNumRows(), A.getNumCols());
            for (size_t i = 0; i < A.getNumRows(); ++i)
                for (size_t j = 0; j < A.getNumCols(); ++j)
                    result(i, j) = A(i, j) * s;
            return result;
        }

        //-----------------------------------------------------------------------------------------
        // Outer product
        //-----------------------------------------------------------------------------------------

        /**
         * @brief The outer product u v^T, whose element (i, j) is u_i v_j.
         *
         * The result is not formed until it is assigned to a Matrix or added into one, so
         * that accumulating it costs no allocation. See OuterProduct.
         */
        template <typename T>
        OuterProduct<T> outer(const Vector<T> &u, const Vector<T> &v)
        {
            return OuterProduct<T>(u, v);
        }

    } // namespace math
} // namespace sw
