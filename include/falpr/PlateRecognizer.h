/**
 * @file PlateRecognizer.h
 * @brief License plate recognition using deep learning models.
 */

#ifndef PLATE_RECOGNIZER_H
#define PLATE_RECOGNIZER_H

#include "FALPR_Shared.h"
#include <opencv2/dnn.hpp>

namespace falpr {

/**
 * @class PlateRecognizer
 * @brief Recognizes characters from license plates using deep learning.
 */
class FALPR_EXPORT PlateRecognizer
{
public:
    /**
     * @enum Country
     * @brief Enum representing supported countries for license plate recognition.
     */
    enum Country {
        Uz = 0 ///< Uzbekistan
    };

    /**
     * @struct Char
     * @brief Represents a recognized character in the license plate.
     */
    struct Char {
        char label;        ///< Recognized character.
        float confidence;  ///< Confidence score of recognition.
        cv::Rect boundingBox; ///< Bounding box of the character.

        /**
         * @brief Outputs character recognition details.
         * @param out Output stream.
         * @param c Character to print.
         * @return Output stream with character details.
         */
        friend std::ostream &operator<<(std::ostream &out, const Char &c) {
            out << c.label << " --> " << c.confidence;
            return out;
        }

        /// Comparison operators for sorting characters based on x-coordinate.
        inline bool operator < (const Char &c) { return this->boundingBox.x < c.boundingBox.x; }
        inline bool operator > (const Char &c) { return this->boundingBox.x > c.boundingBox.x; }
        inline bool operator <= (const Char &c) { return this->boundingBox.x <= c.boundingBox.x; }
        inline bool operator >= (const Char &c) { return this->boundingBox.x >= c.boundingBox.x; }
    };

    /**
     * @struct License
     * @brief Represents a recognized license plate.
     */
    struct License {
        Country country;             ///< Country of the license plate.
        float totalConfidence;       ///< Total confidence score for the plate.
        std::string license;         ///< Recognized license plate number.
        std::vector<Char> characters; ///< List of recognized characters.
    };

    /**
     * @brief Constructor for PlateRecognizer.
     * @param threshold Confidence threshold for recognition.
     * @param modelPath Path to the deep learning model.
     */
    PlateRecognizer(const float &threshold, const std::string &modelPath);

    /**
     * @brief Recognizes a license plate from an input image.
     * @param image Input image (cv::Mat).
     * @return Recognized license plate information.
     */
    License recognize(const cv::Mat &image);

private:
    /**
     * @brief Runs inference on the input image.
     * @param input Cropped license plate image.
     * @return Recognized characters.
     */
    std::vector<Char> runInference(const cv::Mat &input);

private:
    cv::dnn::Net dnn_;          ///< Deep learning network.
    cv::Size2f modelSize_;      ///< Model input size.

    float scoreThreshold_ = 0.75; ///< Confidence score threshold.
    float nmsThreshold_ = 0.75;   ///< Non-maximum suppression threshold.

    /// List of supported characters.
    const std::vector<char> classes_{
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z'
    };
};

} // namespace falpr

#endif // PLATE_RECOGNIZER_H