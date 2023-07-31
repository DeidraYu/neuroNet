#pragma once

#include <execution>
#include <initializer_list>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace sw
{
    template <class T>
    concept arithmetic = std::is_arithmetic_v<T>;
    // concept arithmetic = std::integral<T> || std::floating_point<T>;

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

        void operator+=(const T rhs)
        {
            std::transform(std::execution::par_unseq, begin(), end(), begin(), [rhs](const T &element)
                           { return element + rhs; });
        }

        void operator-=(const T rhs)
        {
            std::transform(std::execution::par_unseq, begin(), end(), begin(), [rhs](const T &element)
                           { return element - rhs; });
        }

        // We require the right hand side to be of the same type as the vector entries already are.
        void operator*=(const T rhs)
        {
            std::transform(std::execution::par_unseq, begin(), end(), begin(), [rhs](const T &element)
                           { return element * rhs; });
        }

        void operator+=(const VectorView<T> &rhs)
        {
            std::transform(std::execution::par_unseq, begin(), end(), rhs.cbegin(), begin(), std::plus<T>());
        }

        void operator-=(const VectorView<T> &rhs)
        {
            std::transform(std::execution::par_unseq, begin(), end(), rhs.cbegin(), begin(), std::minus<T>());
        }

        // We require the right hand side to be of the same type as the vector entries already are.
        // Warning: Note that *= is performing an element-wise / point-wise multiplication. This might
        //          not be what is expected because the typical * performs a standard vector multiplication.
        void operator*=(const VectorView<T> &rhs)
        {
            std::transform(std::execution::par_unseq, begin(), end(), rhs.cbegin(), begin(), std::multiplies<T>());
        }

        // Comparison operator ==
        bool operator==(const VectorView<T> &other) const
        {
            return (*m_pVec) == (*other.m_pVec);
        }

        bool operator!=(const VectorView<T> &other) const
        {
            return (*m_pVec) != (*other.m_pVec);
        }

        // Overload the << operator for output
        friend std::ostream &operator<<(std::ostream &os, const Vector<T> &vec)
        {
            os << vec.toString();
            return os;
        }

        template <typename U>
        Vector<typename std::common_type<T, U>::type> point_mult(const VectorView<U> &rhs) const;

        // in place operation for:
        // u = u + a*v
        void updateWithScaledVector(const T scalar, const VectorView<T> &rhs)
        {
            // std::transform(std::execution::par_unseq, cbegin(), cend(), rhs.cbegin(), begin(), [scalar](const T &vecElement, const T &vecElement_other)
            //                { return vecElement + scalar * vecElement_other; });

            // std::transform(std::execution::par_unseq, begin(), end(), rhs.cbegin(), [scalar](T &thisVecElement, const T &otherVecElement)
            //                { thisVecElement += scalar * otherVecElement; }); // does not build. std::transform requires return statement

            std::for_each(std::execution::par_unseq, begin(), end(), [&](T &thisVecElement)
                          {
        const T & otherVecElement  = rhs[static_cast<int>(&thisVecElement - &((*m_pVec)[0]))];
        // Your operation here, for example, print the sum of each pair
        thisVecElement += static_cast<T>(scalar * otherVecElement); });
        }

        template <typename U>
        Matrix<typename std::common_type<T, U>::type> outer(const VectorView<U> &rhs) const;

        template <typename U>
        Vector<typename std::common_type<T, U>::type> operator+(const VectorView<U> &rhs) const;

        template <typename U>
            requires arithmetic<U>
        Vector<typename std::common_type<T, U>::type> operator+(const U &rhs) const;

        template <typename U>
        Vector<typename std::common_type<T, U>::type> operator-(const VectorView<U> &rhs) const;

        void fill(T value)
        {
            std::fill(begin(), end(), value);
        }

        template <typename U>
            requires arithmetic<U>
        auto operator-(const U &rhs) const;

        typename std::vector<T>::size_type size() const;

        void resize(size_t newSize)
        {
            m_pVec->resize(newSize);
        }

        // Iterators
        typename std::vector<T>::iterator begin() { return m_pVec->begin(); }
        typename std::vector<T>::iterator end() { return m_pVec->end(); }

        // Constant iterators
        auto cbegin() const { return m_pVec->cbegin(); }
        auto cend() const { return m_pVec->cend(); }

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

        Vector(std::vector<T>::size_type sz, T initialValue);

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

                for (uint32_t i = 0; i < sz; ++i)
                {
                    randVector[i] = static_cast<T>(dis(gen));
                }
            }
            else if constexpr (std::is_floating_point_v<T>)
            {
                // std::uniform_real_distribution<T> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
                std::uniform_real_distribution<T> dis(min, max);
                for (uint32_t i = 0; i < sz; ++i)
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

    template <typename T>
    Vector<T>::Vector(std::vector<T>::size_type sz, T initialValue) : m_vecStorage(sz, initialValue)
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
    Vector<T>::Vector(const Vector<T> &other) : VectorView<T>(other), m_vecStorage{other.m_vecStorage}
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
        using CommonType = typename std::common_type<T, U>::type;

        CommonType result{CommonType{0}};

        for (uint32_t i = 0; i < (*m_pVec).size(); ++i)
        {
            result += static_cast<CommonType>(static_cast<CommonType>((*m_pVec)[i]) * static_cast<CommonType>(rhs[i]));
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

        for (uint32_t i = 0; i < size(); ++i)
        {
            resultVec.push_back(static_cast<CommonType>((*m_pVec)[i]) * static_cast<CommonType>(rhs));
        }

        return sw::Vector<CommonType>(resultVec);
    }

    template <typename T>
    template <typename U>
    Vector<typename std::common_type<T, U>::type> VectorView<T>::point_mult(const VectorView<U> &rhs) const
    {
        using CommonType = typename std::common_type<T, U>::type;

        Vector<CommonType> result(size());

        for (uint32_t i = 0; i < (*m_pVec).size(); ++i)
        {
            result[i] = (*m_pVec)[i] * static_cast<CommonType>(rhs[i]);
        }
        return result;
    }

    template <typename T>
    template <typename U>
    Matrix<typename std::common_type<T, U>::type> VectorView<T>::outer(const VectorView<U> &rhs) const
    {
        using CommonType = typename std::common_type<T, U>::type;
        Matrix<CommonType> A(static_cast<uint32_t>(size()), static_cast<uint32_t>(rhs.size()));
        for (uint32_t r = 0; r < size(); ++r)
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

        for (uint32_t i = 0; i < (*m_pVec).size(); ++i)
        {
            result[i] = (*m_pVec)[i] + rhs[i];
        }
        return result;
    }

    template <typename T>
    template <typename U>
        requires arithmetic<U>
    Vector<typename std::common_type<T, U>::type> VectorView<T>::operator+(const U &rhs) const
    {
        Vector<typename std::common_type<T, U>::type> result(size());

        for (uint32_t i = 0; i < (*m_pVec).size(); ++i)
        {
            result[i] = (*m_pVec)[i] + rhs;
        }
        return result;
    }

    template <typename T>
    template <typename U>
        requires arithmetic<U>
    auto VectorView<T>::operator-(const U &rhs) const
    {
        Vector<typename std::common_type<T, U>::type> result(size());

        for (uint32_t i = 0; i < (*m_pVec).size(); ++i)
        {
            result[i] = (*m_pVec)[i] - rhs;
        }
        return result;
    }

    template <typename T>
    template <typename U>
    Vector<typename std::common_type<T, U>::type> VectorView<T>::operator-(const VectorView<U> &rhs) const
    {
        Vector<typename std::common_type<T, U>::type> result(size());

        for (uint32_t i = 0; i < (*m_pVec).size(); ++i)
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
            std::snprintf(buffer, 16, "%7.3f", static_cast<double>(num));
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
        for (uint32_t i = 0; i < size() - 1; ++i)
        {
            str = str + num2string((*m_pVec)[i]) + ", ";
        }
        str = str + num2string((*m_pVec)[(*m_pVec).size() - 1]);
        str += ")";
        return str;
    }

} // namespace sw

/*****************************************************************
 * Vector operations where the left hand side (lhs) is a scalar. *
 ****************************************************************/
template <sw::arithmetic T, typename U>
auto operator*(T lhs, const sw::VectorView<U> &rhs)
{
    return rhs * lhs;
}

template <sw::arithmetic T, typename U>
auto operator+(const T lhs, sw::VectorView<U> &rhs)
{
    return rhs + lhs;
}

template <sw::arithmetic T, typename U>
auto operator-(const T lhs, sw::VectorView<U> &rhs)
{
    using CommonType = typename std::common_type<T, U>::type;

    sw::Vector<CommonType> resultVec(rhs.size());

    for (uint32_t i = 0; i < rhs.size(); ++i)
    {
        resultVec[i] = lhs - rhs[i];
    }

    return resultVec;
}