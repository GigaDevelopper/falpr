/**
 * @file VehicleDetector.h
 * @brief Vehicle detection using deep learning models.
 */

#ifndef VEHICLE_DETECTOR_H
#define VEHICLE_DETECTOR_H

#include "FALPR_Shared.h"
#include <opencv2/dnn.hpp>

namespace falpr {

/**
 * @class VehicleDetector
 * @brief Detects vehicles in images using a deep learning model.
 */
class FALPR_EXPORT VehicleDetector
{
public:
    /**
     * @enum Type
     * @brief Enum representing different vehicle types.
     */
    enum Type {
        BUS = 0,   ///< Bus
        CAR,       ///< Car
        MOTOR,     ///< Motorcycle
        TRUCK,     ///< Truck
        VAN        ///< Van
    };

    /**
     * @struct Vehicle
     * @brief Structure representing a detected vehicle.
     */
    struct Vehicle {
        Type type;           ///< Type of the detected vehicle.
        float confidence;    ///< Confidence score of detection.
        cv::Rect boundingBox; ///< Bounding box of the detected vehicle.
    };

    /**
     * @enum ModelSize
     * @brief Enum representing different model sizes for inference.
     */
    enum ModelSize {
        Nano = 0,   ///< 256x256 model size.
        Small,      ///< 320x320 model size.
        Medium,     ///< 384x384 model size.
        Big,        ///< 416x416 model size.
        Large       ///< 416x416 model size.
    };

    /**
     * @brief Constructor for VehicleDetector.
     * @param modelPath Path to the deep learning model.
     * @param mSize Model size (default: Medium).
     */
    VehicleDetector(const std::string &modelPath,
                    const VehicleDetector::ModelSize &mSize = VehicleDetector::Medium);

    /**
     * @brief Detects vehicles in an input image.
     * @param image Input image (cv::Mat).
     * @return Vector of detected vehicles.
     */
    std::vector<Vehicle> detect(const cv::Mat &image);

private:
    cv::dnn::Net dnn_;         ///< Deep learning network.
    cv::Size2f modelSize_;     ///< Model input size.

    float scoreThreshold_ = 0.8; ///< Confidence score threshold.
    float nmsThreshold_  = 0.8;  ///< Non-maximum suppression threshold.

    std::vector<std::string> classes_{"Bus", "Car", "Motor", "Truck", "Van"}; ///< Vehicle class labels.
};

} // namespace falpr

#endif // VEHICLE_DETECTOR_H
