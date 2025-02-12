#ifndef PLATE_DETECTOR_H
#define PLATE_DETECTOR_H

#include "FALPR_Shared.h"

#include <opencv2/dnn.hpp>

namespace falpr {
class FALPR_EXPORT PlateDetector
{
public:
    enum ModelSize {
        Nano = 0,  // 256x256
        Small ,    // 320x320
        Medium,    // 384x384
        Big,       // 448x448
        Large      // 640x640
    };

    struct Plate {
        float confidence;
        std::vector<cv::Point2f> keypoints;
    };

    PlateDetector(const ModelSize &modelSize,
                  const std::string &modelPath);

    // Detect keypoints of license plates
    std::vector<Plate> detect(const cv::Mat &image);

    // Crop detected plate
    static cv::Mat cropPlate(const cv::Mat &image, const Plate &plate);

private:
    // Prediction parametres
    struct PreParams{
        float ratio = 1.0f;
        float dw = 0.0f;
        float dh = 0.0f;
        float height = 0;
        float width = 0;
    };

    // rescale image and ready to modelSize
    PreParams preprocces(const cv::Mat &in, cv::Mat &out);

private:
    cv::dnn::Net dnn_;
    cv::Size2f modelSize_;

    float scoreThreshold_ = 0.4;
    float nmsThreshold_ = 0.5;
};
} // walpr

#endif // PLATE_DETECTOR_H
