//
// Created by azmiddin on 2/12/25.
//

#ifndef FALPR_H
#define FALPR_H
#include "FALPR_Shared.h"

#include "PlateDetector.h"
#include "PlateRecognizer.h"
#include "VehicleDetector.h"

namespace falpr {
class FALPR_EXPORT FALPR
{
public:
  struct Result {
    VehicleDetector::Vehicle vehicle;
    PlateDetector::Plate plate;
    PlateRecognizer::License license;
  };

  FALPR(const float &charConfidence,
        const float &overallConfidence,
        const PlateDetector::ModelSize &modelSize,
        const std::string &modelPath);

  // recognize from BGR image using OpenCV Mat
  std::vector<Result> recognize(const cv::Mat &image);

  // Draw plate on image
  void drawResult(cv::Mat &image,
                  const Result &result,
                  const cv::Scalar &color);

private:
  PlateDetector plateDetector_;
  PlateRecognizer plateRecognizer_;
  VehicleDetector vehicleDetector_;

  float overallConfidence_ = 0.85;
};
}
#endif //FALPR_H
