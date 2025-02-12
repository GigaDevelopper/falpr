#ifndef VEHICLE_DETECTOR_H
#define VEHICLE_DETECTOR_H

#include "FALPR_Shared.h"

#include <opencv2/dnn.hpp>

namespace falpr {
class FALPR_EXPORT VehicleDetector
{
public:
    enum Type {
        BUS = 0,
        CAR,
        MOTOR,
        TRUCK,
        VAN
    };

    //vehicle detect struct
    struct Vehicle{
        Type type;
        float confidence;
        cv::Rect boundingBox;
    };

    enum ModelSize {
        Nano = 0,   // 256x256
        Small,      // 320x320
        Medium,     // 384x384
        Big,        // 416x416
        Large       // 416x416
    };

    VehicleDetector(const std::string &modelPath,
                    const VehicleDetector::ModelSize &mSize = VehicleDetector::Medium);

    // detect Vehicle
    std::vector<Vehicle> detect(const cv::Mat &image);

private:
    cv::dnn::Net dnn_;
    cv::Size2f modelSize_;

    float scoreThreshold_ = 0.8;
    float nmsThreshold_  = 0.8;

    std::vector<std::string> classes_{"Bus", "Car", "Motor", "Truck", "Van"};
};
} // walpr

#endif // VEHICLE_DETECTOR_H
