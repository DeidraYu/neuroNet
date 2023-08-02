#include <cinttypes>
#include <cstdio>

#include "Utils.hpp"
// #include "Matrix.hpp"
#include "math/Vector.hpp"

int sw::verbosity = 1;

void printNumber(std::vector<uint8_t> &vec)
{
    for (int y = 0; y < 28; y += 2)
    {
        for (int x = 0; x < 28; ++x)
        {
            // uint8_t pixelValue = dataset.training_images[++index];
            uint8_t pix_top = vec[x + 28 * y];
            uint8_t pix_bot = vec[x + 28 * (y + 1)];

            if ((pix_top < 128) && (pix_bot < 128))
            {
                printf(" ");
            }

            if ((pix_top < 128) && (pix_bot >= 128))
            {
                printf("%c", 0xDC);
            }

            if ((pix_top >= 128) && (pix_bot < 128))
            {
                printf("%c", 0xDF);
            }

            if ((pix_top >= 128) && (pix_bot >= 128))
            {
                printf("%c", 0xDB);
            }
        }
        printf("\n");
    }
}

void printProgress(int part, int total, std::string labelString)
{
    if (sw::verbosity < 1)
    {
        return;
    }

    static int prevPercentage = 0;
    const int width = 80;
    const int progressWidth = width - 2; // -2 because we don't count the [] symbols as progress
    int nDone = progressWidth * part / total;
    int percentage = int(100.0f * float(part) / float(total) + 0.5f);

    if (percentage == prevPercentage)
    {
        return;
    }

    char str[width + 1];
    for (int i = 0; i < progressWidth; ++i)
    {
        i < nDone ? str[i + 1] = '#' : str[i + 1] = ' ';
    }
    str[0] = '[';
    str[width - 1] = ']';
    str[width] = '\0';
    printf("\r%s  (%d%%) %s  ", labelString.c_str(), percentage, str);
    fflush(stdout);
    prevPercentage = percentage;
}

void printSW(sw::math::Vector<float> &vec)
{
    for (int y = 0; y < 28; y += 2)
    {
        for (int x = 0; x < 28; ++x)
        {
            // uint8_t pixelValue = dataset.training_images[++index];
            float pix_top = vec[x + 28 * y];
            float pix_bot = vec[x + 28 * (y + 1)];

            if ((pix_top < 0.5f) && (pix_bot < 0.5f))
            {
                printf(" ");
            }

            if ((pix_top < 0.5f) && (pix_bot >= 0.5f))
            {
                printf("%c", 0xDC);
            }

            if ((pix_top >= 0.5f) && (pix_bot < 0.5f))
            {
                printf("%c", 0xDF);
            }

            if ((pix_top >= 0.5f) && (pix_bot >= 0.5f))
            {
                printf("%c", 0xDB);
            }
        }
        printf("\n");
    }
}

Vector<float> oneHotEncode(int value, int numClasses)
{
    Vector<float> encodedVector(numClasses);
    if (value >= 0 && value < numClasses)
    {
        encodedVector[value] = 1.0f;
    }
    return encodedVector;
}

namespace sw
{
    namespace prof
    {
        // Implementations for Counter
        Counter::Counter() : m_time_ns(0), m_count(0) {}
        Counter::Counter(uint64_t time_ns) : m_time_ns(time_ns), m_count(1) {}
        void Counter::update(uint64_t time_ns)
        {
            m_time_ns += time_ns;
            ++m_count;
        }

        uint64_t Counter::count() const { return m_count; }
        uint64_t Counter::time_ns() const { return m_time_ns; }

        float Counter::time_us() const { return m_time_ns / 1'000.0f; }
        float Counter::time_ms() const { return m_time_ns / 1'000'000.0f; }
        float Counter::time_s() const { return m_time_ns / 1'000'000'000.0f; }

        float Counter::mean_ns() const { return static_cast<float>(m_time_ns) / m_count; }
        float Counter::mean_us() const { return static_cast<float>(m_time_ns) / m_count / 1'000.0f; }
        float Counter::mean_ms() const { return static_cast<float>(m_time_ns) / m_count / 1'000'000.0f; }
        float Counter::mean_s() const { return static_cast<float>(m_time_ns) / m_count / 1'000'000'000.0f; }

        // Implementations for Times
        void Times::update(const std::string name, uint64_t time_ns)
        {
            if (m_times.find(name) != m_times.end())
            {
                m_times[name].update(time_ns);
            }
            else
            {
                m_times[name] = Counter(time_ns);
            }
        }

        const Counter &Times::getCounter(const std::string name)
        {
            return m_times[name];
        }

        std::string Times::toString(int nEpochs)
        {
            std::string str = "";
            const size_t N = 256;
            char line[N];

            const int nameWidth = 40;
            const int countWidth = 8;
            const int totalWidth = 10;
            const int meanWidth = 14;

            snprintf(line, N, "%-*s    %-*s    %-*s    %-*s    %-*s\n", nameWidth, "Name", countWidth, "Count", totalWidth, "tot. [ms]", totalWidth, "per epoch [ms]", meanWidth, "per call [us]");
            str += std::string(line);

            for (auto &pair : m_times)
            {
                const std::string &name = pair.first;
                const Counter &counter = pair.second;
                snprintf(line, N, "%-*s    %8" PRIu64 "    %10.2f    %10.2f    %14.2f\n", nameWidth, name.c_str(), counter.count(), counter.time_ms(), counter.time_ms() / nEpochs, counter.mean_us());
                str += std::string(line);
            }
            return str;
        }

        std::unordered_map<std::string, Counter> Times::m_times;

        // Implementations for Measure
        Measure::Measure(std::string name) : m_name(name)
        {
            m_start = std::chrono::steady_clock::now();
        }

        Measure::~Measure()
        {

            m_end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start);

            m_time_ns = duration.count();
            Times::update(m_name, m_time_ns);
        }

    } // namespace prof
} // namespace sw
