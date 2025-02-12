/**
 * @file PlateDetector.h
 * @brief License plate detection using deep learning models.
 */

#ifndef PLATE_DETECTOR_H
#define PLATE_DETECTOR_H

#include "FALPR_Shared.h"
#include <opencv2/dnn.hpp>

namespace falpr {

/**
 * @class PlateDetector
 * @brief Detects license plates in images using deep learning.
 */
class FALPR_EXPORT PlateDetector
{
public:
    /**
     * @enum ModelSize
     * @brief Enum representing different model sizes for detection.
     */
    enum ModelSize {
        Nano = 0,  ///< 256x256 model input size.
        Small,     ///< 320x320 model input size.
        Medium,    ///< 384x384 model input size.
        Big,       ///< 448x448 model input size.
        Large      ///< 640x640 model input size.
    };

    /**
     * @struct Plate
     * @brief Represents a detected license plate.
     */
    struct Plate {
        float confidence;                  ///< Confidence score of the detection.
        std::vector<cv::Point2f> keypoints; ///< Key points of the detected plate.
    };

    /**
     * @brief Constructs a PlateDetector instance.
     * @param modelSize The size of the detection model.
     * @param modelPath The path to the trained model.
     */
    PlateDetector(const ModelSize &modelSize, const std::string &modelPath);

    /**
     * @brief Detects license plates in the given image.
     * @param image Input image (cv::Mat).
     * @return A vector of detected plates with keypoints.
     */
    std::vector<Plate> detect(const cv::Mat &image);

    /**
     * @brief Crops a detected license plate from the input image.
     * @param image Input image.
     * @param plate Detected plate containing keypoints.
     * @return Cropped license plate image.
     */
    static cv::Mat cropPlate(const cv::Mat &image, const Plate &plate);

private:
    /**
     * @struct PreParams
     * @brief Preprocessing parameters for image resizing.
     */
    struct PreParams {
        float ratio = 1.0f;  ///< Scaling ratio.
        float dw = 0.0f;     ///< Padding width.
        float dh = 0.0f;     ///< Padding height.
        float height = 0;    ///< Original image height.
        float width = 0;     ///< Original image width.
    };

    /**
     * @brief Preprocesses an image for model input.
     * @param in Input image.
     * @param out Preprocessed output image.
     * @return Preprocessing parameters.
     */
    PreParams preprocces(const cv::Mat &in, cv::Mat &out);

private:
    cv::dnn::Net dnn_;      ///< Deep learning network for plate detection.
    cv::Size2f modelSize_;  ///< Model input size.

    float scoreThreshold_ = 0.4; ///< Confidence threshold for detection.
    float nmsThreshold_ = 0.5;   ///< Non-maximum suppression threshold.
};

} // namespace falpr

#endif // PLATE_DETECTOR_H
