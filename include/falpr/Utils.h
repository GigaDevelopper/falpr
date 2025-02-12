
#ifndef UTILS_H
#define UTILS_H

#include "FALPR_Shared.h"

#include <opencv2/opencv.hpp>
/**
 * @file utils.h
 * @brief Utility functions for FALPR.
 */

namespace falpr::utils {

/**
 * @brief Clamps a value within the specified range.
 * @param val Input value.
 * @param min Minimum allowed value.
 * @param max Maximum allowed value.
 * @return Clamped value.
 */
FALPR_EXPORT float clamp(float val, float min, float max);

/**
 * @brief Finds the minimum X and Y coordinates from a set of points.
 * @param kps Vector of keypoints.
 * @return Pair of (minX, minY).
 */
FALPR_EXPORT std::pair<float,float> getMinXYPoints(const std::vector<cv::Point2f> &kps);

/**
 * @brief Finds the maximum X and Y coordinates from a set of points.
 * @param kps Vector of keypoints.
 * @return Pair of (maxX, maxY).
 */
FALPR_EXPORT std::pair<float,float> getMaxXYPoints(const std::vector<cv::Point2f> &kps);

/**
 * @brief Computes the center point of a given set of points.
 * @param points Vector of points.
 * @return The center point.
 */
FALPR_EXPORT cv::Point2f center(const std::vector<cv::Point2f> &points);

/**
 * @brief Rotates an image around a given center point.
 * @param image Input image.
 * @param center Center point of rotation.
 * @param angle Rotation angle in degrees.
 * @return Rotated image.
 */
FALPR_EXPORT cv::Mat rotate(const cv::Mat& image, const cv::Point2f& center, double angle);

/**
 * @brief Crops a frame to the specified rectangle.
 * @param frame Input frame.
 * @param rect Cropping rectangle.
 * @return Cropped image.
 */
FALPR_EXPORT cv::Mat cropFrame(const cv::Mat& frame, const cv::Rect& rect);

/**
 * @brief Adjusts brightness automatically using histogram clipping.
 * @param src Source image.
 * @param dst Destination image.
 * @param clipHistPercent Percentage of histogram clipping (default: 0).
 */
FALPR_EXPORT void autoBrightness(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0);

/**
 * @brief Validates if a license plate number follows Uzbekistan's format.
 * @param plateNumber License plate number as a string.
 * @return True if the format is valid, false otherwise.
 */
FALPR_EXPORT bool isValidUz(const std::string &plateNumber);

} // namespace falpr::utils
#endif //UTILS_H
