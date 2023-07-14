#pragma once

#include <cinttypes>

#include "MathUtils.hpp"

namespace sw
{
    template <typename T>
    class Matrix
    {
    public:
        Matrix(uint32_t nRows, uint32_t nCols) : m_nRows(nRows), m_nCols(nCols), m_rows(m_nRows), ref(nRows)
        {
            for (std::vector<T> &row : m_rows)
            {
                row.resize(m_nCols);
            }
        }

        std::vector<T> &operator[](int i)
        {
            return m_rows[i];
        }

        const std::vector<T> &operator[](int i) const
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
                for (size_t c = 0; c < m_nCols; c++)
                {
                    result[static_cast<int>(r)] += m_rows[r][c] * rhs[static_cast<int>(c)];
                }
            }

            return result;
        }

        T sum() const
        {
            T sum{0};

            for (const std::vector<T> &row : m_rows)
            {
                for (const T &element : row)
                {
                    sum += element;
                }
            }

            return sum;
        }

        std::string toString(const std::vector<T> &row)
        {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < row.size(); ++i)
            {
                if (i > 0)
                {
                    oss << ", ";
                }
                oss << row[i];
            }
            oss << "]";
            return oss.str();
        }

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
                oss << toString(m_rows[i]);
            }
            oss << "]";
            return oss.str();
        }

    private:
        uint32_t m_nRows = 0;
        uint32_t m_nCols = 0;

        /// @brief The container of the the matrix data.
        /// The data is an std::vector of rows, where each row is an std::vector of the template type.
        std::vector<std::vector<T>> m_rows;

        uint32_t &ref;
    };

} // namespace sw