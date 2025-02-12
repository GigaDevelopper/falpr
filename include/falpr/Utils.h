
#ifndef UTILS_H
#define UTILS_H

#include "FALPR_Shared.h"

#include <opencv2/opencv.hpp>

namespace falpr::utils {
// clambing return value with image size
FALPR_EXPORT float  clamp(float val, float min, float max);

// matrix operations
FALPR_EXPORT std::pair<float,float> getMinXYPoints(const std::vector<cv::Point2f> &kps);
FALPR_EXPORT std::pair<float,float> getMaxXYPoints(const std::vector<cv::Point2f> &kps);
FALPR_EXPORT cv::Point2f center(const std::vector<cv::Point2f> &points);
FALPR_EXPORT cv::Mat rotate(const cv::Mat& image, const cv::Point2f& center, double angle);

// image enhancements
FALPR_EXPORT cv::Mat cropFrame(const cv::Mat& frame, const cv::Rect& rect);
FALPR_EXPORT void autoBrightness(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0);

// country pattern validations
FALPR_EXPORT bool isValidUz(const std::string &plateNumber);
}
#endif //UTILS_H
