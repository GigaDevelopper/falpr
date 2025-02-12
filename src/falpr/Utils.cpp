
#include "FALPR/Utils.h"

#include "opencv2/imgproc.hpp"

#include <regex>

float falpr::utils::clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

std::pair<float, float> falpr::utils::getMinXYPoints(const std::vector<cv::Point2f> &kps)
{
    float minX = std::ranges::min_element(kps, [](const cv::Point2f &a, const cv::Point2f &b){
                     return a.x < b.x;
                 })->x;
    float minY = std::ranges::min_element(kps, [](const cv::Point2f &a, const cv::Point2f &b){
                     return a.y < b.y;
                 })->y;

    return std::make_pair(minX,minY);
}

std::pair<float, float> falpr::utils::getMaxXYPoints(const std::vector<cv::Point2f> &kps)
{
    float maxX = std::ranges::min_element(kps, [](const cv::Point2f &a, const cv::Point2f &b){
                     return a.x > b.x;
                 })->x;
    float maxY = std::ranges::min_element(kps, [](const cv::Point2f &a, const cv::Point2f &b){
                     return a.y > b.y;
                 })->y;

    return std::make_pair(maxX,maxY);
}

bool falpr::utils::isValidUz(const std::string &plateNumber)
{
    std::vector<std::regex> uzPatterns = {std::regex("^[0][1]\\d{3}[A-Z]{3}$"),
                                          std::regex("^[1-9][0]\\d{3}[A-Z]{3}$"),
                                          std::regex("^[2|7|8|9][5]\\d{3}[A-Z]{3}$"),
                                          std::regex("^[0][1][A-Z]\\d{3}[A-Z]{2}$"),
                                          std::regex("^[1-9][0][A-Z]\\d{3}[A-Z]{2}$"),
                                          std::regex("^[2|7|8|9][5][A-Z]\\d{3}[A-Z]{2}$"),
                                          std::regex("^[0][1][H|M]\\d{6}$"),
                                          std::regex("^[1-9][0][H|M]\\d{6}$"),
                                          std::regex("^[2|7|8|9][5][H|M]\\d{6}$"),
                                          std::regex("^[T]\\d{6}[0][1]$"),
                                          std::regex("^[T]\\d{6}[1-9][0]$"),
                                          std::regex("^[T]\\d{6}[2|7|8|9][5]$"),
                                          std::regex("^[C][M][D]\\d{4}$"),
                                          std::regex("^[D|T|X]\\d{6}$"),
                                          std::regex("^[0][1]\\d{4}[M][V]$"),
                                          std::regex("^[1-9][0]\\d{4}[M][V]$"),
                                          std::regex("^[P][A][A]\\d{3}$"),
                                          std::regex("^[A-Z]\\d{3}[A-Z][A-Z][0][1]$"),
                                          std::regex("^[A-Z]\\d{3}[A-Z][A-Z][1-9][0]$")};

    bool res = false;
    for(const auto &pattern: uzPatterns)
        res = res or std::regex_match(plateNumber, pattern);

    return res;
}

cv::Mat falpr::utils::cropFrame(const cv::Mat &frame, const cv::Rect &rect)
{
    cv::Rect intersection = rect & cv::Rect(0, 0, frame.cols, frame.rows);
    if (intersection.area() <= 0) {
        return {};
    }

    cv::Mat croppedFrame = frame(intersection);
    return croppedFrame.clone();
}

void falpr::utils::autoBrightness(const cv::Mat &src, cv::Mat &dst, float clipHistPercent)
{
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

  constexpr int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.type() == CV_8UC4) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    if (static_cast<int>(clipHistPercent) == 0)
    {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        cv::calcHist(&gray, 1, nullptr, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }

    float inputRange = maxGray - minGray;

    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

    src.convertTo(dst, -1, alpha, beta);

    // restore alpha channel from source
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = { 3, 3};
        cv::mixChannels(&src, 4, &dst,1, from_to, 1);
    }
}

cv::Point2f falpr::utils::center(const std::vector<cv::Point2f> &points)
{
    float sum_x = 0;
    float sum_y = 0;
    for (const auto &point : points) {
        sum_x += point.x;
        sum_y += point.y;
    }
    return cv::Point2f{static_cast<float>(sum_x / 4.0), static_cast<float>(sum_y / 4.0)};
}

cv::Mat falpr::utils::rotate(const cv::Mat &image, const cv::Point2f &center, double angle)
{
    cv::Mat rotation_matrix = getRotationMatrix2D(center, angle, 1.0);

    cv::Mat rotated_image;
    warpAffine(image, rotated_image, rotation_matrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return rotated_image;
}
