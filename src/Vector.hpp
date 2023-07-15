#pragma once

#include <random>
#include <vector>

#include <type_traits>
#include <string>
#include <initializer_list>

namespace sw
{
    // forward declaration of the matrix class
    template <typename T>
    class Matrix;

    template <typename T>
    class Vector;

    /**
     * @brief Class to put a vector view on an std::vector.
     *
     * Let \f$v\f$ be a vector.
     *
     */
    template <typename T>
    class VectorView
    {
    public:
        VectorView() = default;

        VectorView(std::vector<T> *pVec);

        T &operator[](int i);

        const T &operator[](int i) const;

        template <typename U>
        typename std::common_type<T, U>::type operator*(const VectorView<U> &rhs) const;

        template <typename U>
        std::enable_if_t<std::is_arithmetic_v<U>,
                         sw::Vector<typename std::common_type<T, U>::type>>
        operator*(const U &rhs) const;

        template <typename U>
        Vector<typename std::common_type<T, U>::type> point_mult(const VectorView<U> &rhs) const;

        template <typename U>
        Matrix<typename std::common_type<T, U>::type> outer(const VectorView<U> &rhs) const;

        template <typename U>
        Vector<typename std::common_type<T, U>::type> operator+(const VectorView<U> &rhs) const;

        template <typename U>
        Vector<typename std::common_type<T, U>::type> operator-(const VectorView<U> &rhs) const;

        typename std::vector<T>::size_type size() const;

        void resize(size_t newSize)
        {
            (*m_pVec).resize(newSize);
        }

        std::string toString() const;

        std::string num2string(T num) const;

        std::vector<T> &getStdVector()
        {
            return (*m_pVec);
        }

        uint64_t argmax()
        {
            auto maxElement = std::max_element((*m_pVec).begin(), (*m_pVec).end());
            uint64_t index = std::distance((*m_pVec).begin(), maxElement);
            return index;
        }

    public: // change to protected
        std::vector<T> *m_pVec = nullptr;
    };

    template <typename T>
    class Vector : public VectorView<T>
    {
    public:
        Vector() = default;

        Vector(std::vector<T> vec);

        Vector(std::vector<T>::size_type sz);

        Vector(std::initializer_list<T> initializerList);

        // Copy constructor
        Vector(const Vector<T> &vec);

        // Copy assignment operator
        Vector<T> &operator=(const Vector<T> &other);

        /**
         * @brief Create a random matrix with values uniformly distributed between min and max (inclusive).
         */
        static Vector<T> rand(std::vector<T>::size_type sz, T min, T max)
        {
            std::random_device rd;  // Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

            Vector<T> randVector(sz);

            if constexpr (std::is_integral_v<T>)
            {
                // For some reason the std does not define the std::uniform_int_distribution<> for the uint8_t type.
                using CommonType = typename std::common_type<T, uint16_t>::type;
                std::uniform_int_distribution<CommonType> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

                for (int i = 0; i < sz; ++i)
                {
                    randVector[i] = dis(gen);
                }
            }
            else if constexpr (std::is_floating_point_v<T>)
            {
                // std::uniform_real_distribution<T> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
                std::uniform_real_distribution<T> dis(min, max);
                for (int i = 0; i < sz; ++i)
                {
                    randVector[i] = dis(gen);
                }
            }
            return randVector;
        }

    private:
        std::vector<T> m_vecStorage; //{7, 8, 9};
    };

    // CONSTRUCTOR:
    // VectorView
    template <typename T>
    VectorView<T>::VectorView(std::vector<T> *pVec) : m_pVec{pVec} {}

    // Vector
    template <typename T>
    Vector<T>::Vector(std::vector<T> vec) : m_vecStorage{vec}
    {
        this->m_pVec = &m_vecStorage;
    }

    template <typename T>
    Vector<T>::Vector(std::vector<T>::size_type sz) : m_vecStorage(sz)
    {
        this->m_pVec = &m_vecStorage;
    }

    // template <typename T>
    // Vector<T>::Vector(std::initializer_list<T> initializerList) : m_vecStorage{initializerList}, VectorView<T>{m_vecStorage} {}

    template <typename T>
    Vector<T>::Vector(std::initializer_list<T> initializerList) : m_vecStorage{initializerList}
    {
        this->m_pVec = &m_vecStorage;
    }

    template <typename T>
    Vector<T>::Vector(const Vector<T> &other) : m_vecStorage{other.m_vecStorage}
    {
        this->m_pVec = &m_vecStorage;
    }

    template <typename T>
    Vector<T> &Vector<T>::operator=(const Vector<T> &other)
    {
        if (this == &other)
        {
            return *this;
        }

        m_vecStorage = other.m_vecStorage;
        this->m_pVec = &m_vecStorage;

        return *this;
    }

    // NON-CONSTRUCTOR VectorView
    template <typename T>
    T &VectorView<T>::operator[](int i)
    {
        return (*m_pVec)[i];
    }

    template <typename T>
    const T &VectorView<T>::operator[](int i) const
    {
        return (*m_pVec)[i];
    }

    template <typename T>
    template <typename U>
    typename std::common_type<T, U>::type VectorView<T>::operator*(const VectorView<U> &rhs) const
    {
        typename std::common_type<T, U>::type result{}; // = 0;

        for (int i = 0; i < (*m_pVec).size(); ++i)
        {
            result += (*m_pVec)[i] * rhs[i];
        }

        return result;
    }

    template <typename T>
    template <typename U>
    std::enable_if_t<std::is_arithmetic_v<U>,
                     sw::Vector<typename std::common_type<T, U>::type>>
    VectorView<T>::operator*(const U &rhs) const
    {
        using CommonType = typename std::common_type<T, U>::type;

        std::vector<CommonType> resultVec;
        resultVec.reserve(size());

        for (int i = 0; i < size(); ++i)
        {
            resultVec.push_back((*m_pVec)[i] * rhs);
        }

        return sw::Vector<CommonType>(resultVec);
    }

    template <typename T>
    template <typename U>
    Vector<typename std::common_type<T, U>::type> VectorView<T>::point_mult(const VectorView<U> &rhs) const
    {
        Vector<typename std::common_type<T, U>::type> result(size());

        for (int i = 0; i < (*m_pVec).size(); ++i)
        {
            result[i] = (*m_pVec)[i] * rhs[i];
        }
        return result;
    }

    template <typename T>
    template <typename U>
    Matrix<typename std::common_type<T, U>::type> VectorView<T>::outer(const VectorView<U> &rhs) const
    {
        using CommonType = typename std::common_type<T, U>::type;
        Matrix<CommonType> A(static_cast<uint32_t>(size()), static_cast<uint32_t>(size()));
        for (int r = 0; r < size(); ++r)
        {
            A[r] = (*m_pVec)[r] * rhs;
        }
        return A;
    }

    template <typename T>
    template <typename U>
    Vector<typename std::common_type<T, U>::type> VectorView<T>::operator+(const VectorView<U> &rhs) const
    {
        Vector<typename std::common_type<T, U>::type> result(size());

        for (int i = 0; i < (*m_pVec).size(); ++i)
        {
            result[i] = (*m_pVec)[i] + rhs[i];
        }
        return result;
    }

    template <typename T>
    template <typename U>
    Vector<typename std::common_type<T, U>::type> VectorView<T>::operator-(const VectorView<U> &rhs) const
    {
        Vector<typename std::common_type<T, U>::type> result(size());

        for (int i = 0; i < (*m_pVec).size(); ++i)
        {
            result[i] = (*m_pVec)[i] - rhs[i];
        }
        return result;
    }

    template <typename T>
    typename std::vector<T>::size_type VectorView<T>::size() const
    {
        return (*m_pVec).size();
    }

    template <typename T>
    std::string VectorView<T>::num2string(T num) const
    {
        char buffer[16];
        if constexpr (std::is_integral_v<T>)
        {
            std::snprintf(buffer, 16, "%5d", num);
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            std::snprintf(buffer, 16, "%7.3f", num);
        }

        // else if (std::is_base_of<VectorView<int>, T>::value || std::is_same<T, VectorView<int>>::value)
        // {
        //     // return "+";
        //     return num.toString();
        // }
        // else
        // {
        //     return "-";
        // }

        else
        {
            return num.toString();
        }

        return buffer;
    }

    template <typename T>
    std::string VectorView<T>::toString() const
    {
        std::string str = "(";
        for (int i = 0; i < size() - 1; ++i)
        {
            str = str + num2string((*m_pVec)[i]) + ", ";
        }
        str = str + num2string((*m_pVec)[(*m_pVec).size() - 1]);
        str += ")";
        return str;
    }

} // namespace sw

// Note, we are outside the sw namespace here.
// Overload operator* for float and sc::Vector
template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T>,
                 sw::Vector<typename std::common_type<T, U>::type>>
operator*(T lhs, const sw::VectorView<U> &rhs);

// Note, we are outside the sw namespace here.
// Overload operator* for float and sc::Vector
template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T>,
                 sw::Vector<typename std::common_type<T, U>::type>>

operator*(T lhs, const sw::VectorView<U> &rhs)
{
    using CommonType = typename std::common_type<T, U>::type;

    std::vector<CommonType> resultVec;
    resultVec.reserve(rhs.size());

    for (int i = 0; i < rhs.size(); ++i)
    {
        resultVec.push_back(lhs * rhs[i]);
    }
    // }
    return sw::Vector<CommonType>(resultVec);
}