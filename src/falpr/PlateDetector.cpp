#include "FALPR/PlateDetector.h"

#include "FALPR/Utils.h"

#include <opencv2/imgproc.hpp>

falpr::PlateDetector::PlateDetector(const ModelSize &modelSize, const std::string &modelPath)
{
    switch (modelSize)
    {
    case Nano:{
        dnn_ = cv::dnn::readNetFromONNX(modelPath +"256x256.onnx");
        modelSize_ = cv::Size(256,256);
        break;
    }
    case Small: {
        dnn_ = cv::dnn::readNetFromONNX(modelPath +"320x320.onnx");
        modelSize_ = cv::Size(320,320);
        break;
    }
    case Medium: {
        dnn_ = cv::dnn::readNetFromONNX(modelPath +"384x384.onnx");
        modelSize_ = cv::Size(384,384);
        break;
    }
    case Big: {
        dnn_ = cv::dnn::readNetFromONNX(modelPath +"448x448.onnx");
        modelSize_ = cv::Size(448,448);
        break;
    }
    case Large: {
        dnn_ = cv::dnn::readNetFromONNX(modelPath +"640x640.onnx");
        modelSize_ = cv::Size(640,640);
        break;
    }
    }
}

std::vector<falpr::PlateDetector::Plate> falpr::PlateDetector::detect(const cv::Mat &image)
{
    if(image.empty())
        return std::vector<falpr::PlateDetector::Plate>();

    cv::Mat blob;
    std::vector<cv::Mat> outputs;

    PreParams pp = preprocces(image, blob);

    dnn_.setInput(blob);
    dnn_.forward(outputs, dnn_.getUnconnectedOutLayersNames());

    const int channels = outputs[0].size[2];
    const int anchors = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, anchors);
    cv::Mat output = outputs[0].t();

    std::vector<cv::Rect> bboxList;
    std::vector<float> scoreList;
    std::vector<int> indicesList;
    std::vector<std::vector<cv::Point2f>> kpList;

    for (int i = 0; i < channels; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bbox_ptr = row_ptr;
        auto score_ptr = row_ptr + 4;
        auto kp_ptr = row_ptr + 5;

        float score = *score_ptr;
        if (score > scoreThreshold_) {
            float x = *bbox_ptr++ - pp.dw;
            float y = *bbox_ptr++ - pp.dh;
            float w = *bbox_ptr++;
            float h = *bbox_ptr;

            float x0 = falpr::utils::clamp((x - 0.5f * w) * pp.ratio, 0.f, pp.width);
            float y0 = falpr::utils::clamp((y - 0.5f * h) * pp.ratio, 0.f, pp.height);
            float x1 = falpr::utils::clamp((x + 0.5f * w) * pp.ratio, 0.f, pp.width);
            float y1 = falpr::utils::clamp((y + 0.5f * h) * pp.ratio, 0.f, pp.height);

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;

            bbox.width = (x1 - x0);
            bbox.height = (y1 - y0);

            std::vector<cv::Point2f> kps;
            for (int k = 0; k < 4; k++) {
                float kps_x = (*(kp_ptr + 3 * k) - pp.dw) * pp.ratio;
                float kps_y = (*(kp_ptr + 3 * k + 1) - pp.dh) * pp.ratio;
                // float kps_s = *(kp_ptr + 3 * k + 2);
                kps_x = falpr::utils::clamp(kps_x, 0.f, pp.width);
                kps_y = falpr::utils::clamp(kps_y, 0.f, pp.height);
                kps.emplace_back(kps_x,kps_y);
            }

            bboxList.push_back(bbox);
            scoreList.push_back(score);
            kpList.push_back(kps);
        }
    }

    cv::dnn::NMSBoxes(
        bboxList,
        scoreList,
        scoreThreshold_,
        nmsThreshold_,
        indicesList
        );

    std::vector<Plate> result;
    for(auto &index: indicesList) {
        result.push_back(Plate{scoreList[index], kpList[index]});
    }

    return result;
}

cv::Mat falpr::PlateDetector::cropPlate(const cv::Mat &image, const Plate &plate)
{
    if(image.empty())
        return cv::Mat();

    std::vector<cv::Point2f> points(plate.keypoints);

    float x_collect[4] = {points[0].x, points[1].x, points[2].x, points[3].x};
    float y_collect[4] = {points[0].y, points[1].y, points[2].y, points[3].y};

    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    if(img_crop.empty())
        return cv::Mat();

    for (int i = 0; i < points.size(); i++) {
        points[i].x -= left;
        points[i].y -= top;
    }

    auto  box = plate.keypoints;
    auto wTop = pow(box[0].x - box[1].x, 2) +
                pow(box[0].y - box[1].y, 2);
    auto wBottom =pow(box[2].x - box[3].x, 2) +
                   pow(box[2].y - box[3].y, 2);

    auto hLeft = pow(box[0].x - box[3].x, 2) + pow(box[0].y - box[3].y, 2);
    auto hRight = pow(box[1].x - box[2].x, 2) + pow(box[1].y - box[2].y, 2);

    int img_crop_width = static_cast<int>(sqrt(std::max(wTop,wBottom)));
    int img_crop_height = static_cast<int>(sqrt(std::max(hLeft,hRight)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0].x - 3.f, points[0].y - 3.f);
    pointsf[1] = cv::Point2f(points[1].x + 3.f, points[1].y - 3.f);
    pointsf[2] = cv::Point2f(points[2].x + 3.f, points[2].y + 3.f);
    pointsf[3] = cv::Point2f(points[3].x - 3.f, points[3].y + 3.f);

    // FIXME:  error: (-215:Assertion failed) _src.total() > 0 in function 'warpPerspective'
    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img,ans;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);

    // WARNING: is auto brighnes adjusting is needed?
    falpr::utils::autoBrightness(dst_img, ans, 20.0);
    return ans;
}

falpr::PlateDetector::PreParams falpr::PlateDetector::preprocces(const cv::Mat &in, cv::Mat &out)
{
    const float inp_h  = modelSize_.height * 1.0F;
    const float inp_w  = modelSize_.width * 1.0F;
    float height = in.rows * 1.0F;
    float width  = in.cols * 1.0F;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh)
        cv::resize(in, tmp, cv::Size(padw, padh));
    else
        tmp = in.clone();

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});
    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, modelSize_, cv::Scalar(0, 0, 0), true, false, CV_32F);

    return PreParams{static_cast<float>(1.0/r), dw, dh, height, width};
}
