#ifndef PLATE_RECOGNIZER_H
#define PLATE_RECOGNIZER_H

#include "FALPR_Shared.h"

#include <opencv2/dnn.hpp>

namespace falpr {
class FALPR_EXPORT PlateRecognizer
{
public:
    enum Country {
        Uz = 0
    };

    struct Char {
        char label;
        float confidence;
        cv::Rect boundingBox;

        friend std::ostream &operator<<(std::ostream &out, const Char &c) {
            out << c.label << " -- >" << c.confidence;
            return out;
        }

        inline bool operator < (const Char &c) {
            return this->boundingBox.x < c.boundingBox.x;
        }

        inline bool operator > (const Char &c) {
            return this->boundingBox.x > c.boundingBox.x;
        }

        inline bool operator <= (const Char &c) {
            return this->boundingBox.x <= c.boundingBox.x;
        }

        inline bool operator >= (const Char &c) {
            return this->boundingBox.x >= c.boundingBox.x;
        }
    };

    struct License {
        Country country;
        float totalConfidence;
        std::string license;
        std::vector<Char> characters;
    };

    PlateRecognizer(const float &threshold,
                    const std::string &modelPath);

    // Recognize cropped and alligned license plate
    License recognize(const cv::Mat &image);

private:
    std::vector<Char> runInference(const cv::Mat &input);

private:
    cv::dnn::Net dnn_;
    cv::Size2f modelSize_;

    float scoreThreshold_ = 0.75;
    float nmsThreshold_ = 0.75;

    const std::vector<char> classes_{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                                     'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                                     'u', 'v', 'w', 'x', 'y', 'z'};
};
} // walpr

#endif // PLATE_RECOGNIZER_H
