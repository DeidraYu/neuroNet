#include "Utils.hpp"
#include "Matrix.hpp"

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

int getConsoleWidth()
{
    // Open a pipe to the 'tput cols' command
    FILE *pipe = popen("tput cols", "r");
    if (!pipe)
    {
        // Unable to open pipe, return a default value
        return 80; // Default to 80 columns if tput is not available
    }

    // Read the output of the command into a buffer
    char buffer[128];
    std::string result = "";
    while (!feof(pipe))
    {
        if (fgets(buffer, 128, pipe) != nullptr)
            result += buffer;
    }

    // Close the pipe and get the terminal width as an integer
    pclose(pipe);
    int width = std::stoi(result);
    return width;
}

void printProgress(int part, int total, std::string labelString)
{
    const int width = getConsoleWidth() - 16;
    const int progressWidth = width - 2; // -2 because we don't count the [] symbols as progress
    int nDone = progressWidth * part / total;
    int percentage = int(100.0f * float(part) / float(total) + 0.5f);

    char str[1000];
    for (int i = 0; i < progressWidth; ++i)
    {
        i < nDone ? str[i + 1] = '#' : str[i + 1] = ' ';
    }
    str[0] = '[';
    str[width - 1] = ']';
    str[width] = '\0';
    printf("\r%s  (%d%%) %s", labelString.c_str(), percentage, str);
    fflush(stdout);
}

void printSW(sw::Vector<float> &vec)
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

sw::Vector<float> oneHotEncode(int value, int numClasses)
{
    sw::Vector<float> encodedVector(numClasses);
    if (value >= 0 && value < numClasses)
    {
        encodedVector[value] = 1.0f;
    }
    return encodedVector;
}