#pragma once

#include <vector>
#include <type_traits>
#include <string>
#include <initializer_list>

namespace sw
{

    struct A
    {
        A()
        {
            printf("constructed A\n");
        }
    };

    struct B
    {
        B()
        {
            printf("constructed B\n");
        }
    };

    template <typename T>
    class VectorView
    {
    public:
        A a;
        VectorView() = default;

        VectorView(std::vector<T> vec);

        // VectorView(std::vector<T>::size_type sz);

        // VectorView(std::initializer_list<T> initializerList);

        T &operator[](int i);

        const T &operator[](int i) const;

        template <typename U>
        typename std::common_type<T, U>::type operator*(const VectorView<U> &rhs) const;

        template <typename U>
        VectorView<typename std::common_type<T, U>::type> point_mult(const VectorView<U> &rhs) const;

        typename std::vector<T>::size_type size() const;

        std::string toString() const;

        std::string num2string(T num) const;

    private:
        std::vector<T> &m_vec;
    };

    template <typename T>
    class Vector : public VectorView<T>
    {
    public:
        B b;
        Vector() = default;

        Vector(std::vector<T> vec);

        Vector(std::vector<T>::size_type sz);

        Vector(std::initializer_list<T> initializerList);

        // std::vector<T> &initialize(std::initializer_list<T> initializerList);

    private:
        std::vector<T> m_vecStorage{7, 8, 9};
    };

    template <typename T>
    VectorView<T>::VectorView(std::vector<T> vec) : a{}, m_vec{vec} {}

    template <typename T>
    Vector<T>::Vector(std::vector<T> vec) : m_vecStorage{vec}, VectorView<T>{m_vecStorage} {}

    template <typename T>
    Vector<T>::Vector(std::vector<T>::size_type sz) : m_vecStorage(sz), VectorView<T>{m_vecStorage} {}

    template <typename T>
    Vector<T>::Vector(std::initializer_list<T> initializerList) : /*m_vecStorage{initializerList},*/ VectorView<T>{m_vecStorage} {}

    // template <typename T>
    // std::vector<T> &Vector<T>::initialize(std::initializer_list<T> initializerList)
    // {
    //     m_vecStorage{initializerList};
    //     return m_vecStorage;
    // }

    // template <typename T>
    // Vector<T>::Vector(std::initializer_list<T> initializerList) : b{}, m_vecStorage{initializerList}, VectorView<T>{m_vecStorage} {}

    // template <typename T>
    // Vector<T>::Vector(std::initializer_list<T> initializerList) : b{}, m_vecStorage{initialize(initializerList)} {}

    /*
        template <typename T>
        VectorView<T>::VectorView(std::vector<T>::size_type sz) : m_vec(sz) {}

        template <typename T>
        VectorView<T>::VectorView(std::initializer_list<T> initializerList) : m_vec{initializerList} {}
    */

    template <typename T>
    T &VectorView<T>::operator[](int i)
    {
        return m_vec[i];
    }

    template <typename T>
    const T &VectorView<T>::operator[](int i) const
    {
        return m_vec[i];
    }

    template <typename T>
    template <typename U>
    typename std::common_type<T, U>::type VectorView<T>::operator*(const VectorView<U> &rhs) const
    {
        typename std::common_type<T, U>::type result{}; // = 0;

        for (int i = 0; i < m_vec.size(); ++i)
        {
            result += m_vec[i] * rhs[i];
        }

        return result;
    }

    template <typename T>
    template <typename U>
    VectorView<typename std::common_type<T, U>::type> VectorView<T>::point_mult(const VectorView<U> &rhs) const
    {
        Vector<typename std::common_type<T, U>::type> result(size());

        for (int i = 0; i < m_vec.size(); ++i)
        {
            result[i] = m_vec[i] * rhs[i];
        }
        return result;
    }

    template <typename T>
    typename std::vector<T>::size_type VectorView<T>::size() const
    {
        return m_vec.size();
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
        else
        {
            return "-";
        }
        return buffer;
    }

    template <typename T>
    std::string VectorView<T>::toString() const
    {
        std::string str = "(";
        for (int i = 0; i < size() - 1; ++i)
        {
            str = str + num2string(m_vec[i]) + ", ";
        }
        str = str + num2string(m_vec[m_vec.size() - 1]);
        str += ")";
        return str;
    }

} // namespace sw

// Note, we are outside the sw namespace here.
// Overload operator* for float and sc::Vector
template <typename T>
sw::VectorView<T> operator*(float lhs, const sw::VectorView<T> &rhs);

// Note, we are outside the sw namespace here.
// Overload operator* for float and sc::Vector
template <typename T>
sw::VectorView<T> operator*(float lhs, const sw::VectorView<T> &rhs)
{
    std::vector<T> resultVec;
    resultVec.reserve(rhs.size());

    for (int i = 0; i < rhs.size(); ++i)
    {
        resultVec.push_back(lhs * rhs[i]);
    }

    return sw::VectorView<T>(resultVec);
}