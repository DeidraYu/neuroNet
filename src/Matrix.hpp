#pragma once

#include <cinttypes>

#include "MathUtils.hpp"

namespace sw
{
    template <typename T>
    class Matrix
    {
    public:
        Matrix() = default;

        Matrix(uint32_t nRows, uint32_t nCols) : m_nRows(nRows), m_nCols(nCols), m_rows(m_nRows)
        {
            for (size_t r = 0; r < m_nRows; ++r)
            {
                m_rows[r] = std::move(Vector<T>(m_nCols));
            }
        }

        Vector<T> &operator[](int i)
        {
            return m_rows[i];
        }

        const Vector<T> &operator[](int i) const
        {
            return m_rows[i];
        }

        template <typename U>
        Vector<typename std::common_type<T, U>::type> operator*(const VectorView<U> &rhs) const
        {
            using CommonType = typename std::common_type<T, U>::type;

            Vector<CommonType> result(m_nRows);

            for (size_t r = 0; r < m_nRows; ++r)
            {
                result[static_cast<int>(r)] = m_rows[r] * rhs;
            }

            return result;
        }

        T sum() const
        {
            T sum{0};

            for (const Vector<T> &row : m_rows)
            {
                for (int c = 0; c < row.size(); ++c)
                {
                    sum += row[c];
                }
            }

            return sum;
        }

        // std::string toString(const std::vector<T> &row)
        // {
        //     std::ostringstream oss;
        //     oss << "[";
        //     for (size_t i = 0; i < row.size(); ++i)
        //     {
        //         if (i > 0)
        //         {
        //             oss << ", ";
        //         }
        //         oss << row[i];
        //     }
        //     oss << "]";
        //     return oss.str();
        // }

        std::string toString()
        {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < m_rows.size(); ++i)
            {
                if (i > 0)
                {
                    oss << ", ";
                }
                // oss << toString(m_rows[i]);
                oss << m_rows[i].toString();
            }
            oss << "]";
            return oss.str();
        }

        uint32_t getNumRows()
        {
            return m_nRows;
        }

        uint32_t getNumCols()
        {
            return m_nCols;
        }

        /**
         * @brief Create a random matrix with values uniformly distributed between min and max (inclusive).
         */
        static Matrix<T> rand(uint32_t nRows, uint32_t nCols, T min, T max)
        {
            Matrix<T> randomMatrix(nRows, nCols);
            for (uint32_t r = 0; r < nRows; ++r)
            {
                randomMatrix[r] = Vector<T>::rand(nCols, min, max);
            }
            return randomMatrix;
        }

    private:
        uint32_t m_nRows = 0;
        uint32_t m_nCols = 0;

        /// @brief The container of the the matrix data.
        /// The data is an std::vector of rows, where each row is an std::vector of the template type.

        // std::vector<std::vector<T>> m_rows;
        std::vector<Vector<T>> m_rows;
    };

} // namespace sw