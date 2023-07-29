#pragma once

#include <vector>
#include "math/Vector.hpp"
#include "math/Matrix.hpp"
// #include "Net.hpp"
// #include "Evaluator.hpp"

/**
 * This class describes a single layer of a multi-layer network.
 *
 *
 *                      feed forward layer ℓ
 *                 +---------------------------------------+
 *                 |                                       |
 *           x(ℓ)  |      z(ℓ) = W(ℓ) * x(ℓ) + b(ℓ)        | x(ℓ+1)
 *          ------>|                                       |------>
 *                 |     x(ℓ+1) = y(ℓ) = sigma(z(ℓ))       |
 *                 |                                       |
 *                 +---------------------------------------+
 *
 *
 *
 *
 *                      back propagation layer ℓ
 *                 +---------------------------------------+
 *                 |                                       |
 *                 |  nablaC_b(ℓ) = v(ℓ) .* sigma'(z(ℓ))   |
 *  v(ℓ-1) = u(ℓ)  |                                       | v(ℓ) = u(ℓ+1)
 *          <------|  nablaC_W(ℓ) = nablaC_b(ℓ) * x(ℓ)^T   |<------
 *                 |                                       |
 *                 |  u(ℓ)  = W^T * nablaC_b(ℓ)            |
 *                 |                                       |
 *                 +---------------------------------------+
 */

using namespace sw::math;

class Layer
{
public:
    Layer(uint16_t numInputs, uint16_t numOutputs) : m_numInputs(numInputs),
                                                     m_numOutputs(numOutputs),
                                                     m_W(numOutputs, numInputs, 0.5f),
                                                     m_b(numOutputs, 0.0f),
                                                     m_z(numOutputs, 0.0f),
                                                     m_y(numOutputs, 0.0f),
                                                     m_gamma(numOutputs, 0.0f),
                                                     m_nablaC_b(numOutputs, 0.0f),
                                                     m_nablaC_W(numOutputs, numInputs, 0.0f),
                                                     m_u(numInputs) {}

    void feedForward(const Vector<float> &x)
    {
        m_z = m_W * x + m_b;
        m_y = sigmoid(m_z);
    }

    // call the backProp for miniBatchSize
    void backProp(const Vector<float> &x, const Vector<float> &v, bool updateU = true)
    {
        m_gamma = v * sigmoid_prime(m_z);
        m_nablaC_b += m_gamma;
        m_nablaC_W += outer(m_gamma, x); // (u .* sigma'(z)) x^T

        // Optimization, for Layer0 we must not compute m_u because there is nothing to back propagate to.
        if (updateU == true)
        {
            m_u = m_W.transposeMult(m_gamma);
        }
        m_miniBatchSize++;
    }

    // update the Weight matrix and bias vector
    void update(float eta)
    {
        float scaleFactor = -eta / m_miniBatchSize; // Note the minus sign before eta, such that the following two lines get the +=
        m_W += m_nablaC_W * scaleFactor;            // TODO, implement eta
        m_b += m_nablaC_b * scaleFactor;            // TODO, implement eta

        m_nablaC_b.fill(0.0f);
        m_nablaC_W.fill(0.0f);
        m_miniBatchSize = 0;
    }

    void randomizeWB(float min, float max)
    {
        m_W = Matrix<float>::rand(m_W.getNumRows(), m_W.getNumCols(), min, max);
    }

    constexpr uint16_t size() const { return m_numOutputs; }

    const auto &getW() const { return m_W; }
    const auto &getB() const { return m_b; }

    const auto &getZ() const { return m_z; }
    const auto &getY() const { return m_y; }
    const auto &getU() const { return m_u; }

    const auto &getNablaC_W() const { return m_nablaC_W; }
    const auto &getNablaC_b() const { return m_nablaC_b; }

private:
    void feedforward(Vector<float> &x);

    // float sigmoid(float z)
    // {
    //     return 1.0f / (1.0f + static_cast<float>(exp(-z)));
    // }

    // float sigmoid_prime(float z)
    // {
    //     float s = sigmoid(z);
    //     return (1.0f - s) * s;
    // }

    // Vector<float> sigmoid(sw::Vector<float> z)
    // {
    //     sw::Vector<float> y(z.size());

    //     for (uint32_t i = 0; i < z.size(); ++i)
    //     {
    //         y[i] = sigmoid(z[i]);
    //     }

    //     return y;
    // }

    // Vector<float> sigmoid_prime(sw::Vector<float> z)
    // {
    //     sw::Vector<float> y(z.size());

    //     for (uint32_t i = 0; i < z.size(); ++i)
    //     {
    //         y[i] = sigmoid_prime(z[i]);
    //     }

    //     return y;
    // }

    Vector<float> oneHotEncode(int value, int numClasses);

    uint16_t m_numInputs;
    uint16_t m_numOutputs;

    Matrix<float> m_W;
    Vector<float> m_b;
    Vector<float> m_z;
    Vector<float> m_y;

    Vector<float> m_gamma;
    Vector<float> m_nablaC_b;
    Matrix<float> m_nablaC_W;
    Vector<float> m_u;

    uint16_t m_miniBatchSize{0};
};
