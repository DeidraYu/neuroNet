#include "Utils.hpp"

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
    const int width = 80;
    const int progressWidth = width - 2; // -2 because we don't count the [] symbols as progress
    int nDone = progressWidth * part / total;
    int nTodo = progressWidth - nDone;

    char str[width + 1];
    for (int i = 0; i < progressWidth; ++i)
    {
        i < nDone ? str[i + 1] = '#' : str[i + 1] = ' ';
    }
    str[0] = '[';
    str[width - 1] = ']';
    str[width] = '\0';
    printf("\r%s %s", labelString.c_str(), str);
}