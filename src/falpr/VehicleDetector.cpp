#include "FALPR/VehicleDetector.h"

falpr::VehicleDetector::VehicleDetector(const std::string &modelPath,
                                        const VehicleDetector::ModelSize &mSize)
{
    switch(mSize){
    case Nano:{
        modelSize_ = cv::Size2f(256, 256);
        dnn_ = cv::dnn::readNetFromONNX(modelPath + "256x256v.onnx");
        break;
    }
    case Small:{
        modelSize_ = cv::Size2f(256, 256);
        dnn_ = cv::dnn::readNetFromONNX(modelPath + "256x256v.onnx");
        break;
    }
    case Medium:{
        modelSize_ = cv::Size2f(320, 320);
        dnn_ = cv::dnn::readNetFromONNX(modelPath + "320x320v.onnx");
        break;
    }
    case Big:{
        modelSize_ = cv::Size2f(384, 384);
        dnn_ = cv::dnn::readNetFromONNX(modelPath + "384x384v.onnx");
        break;
    }
    case Large:{
        modelSize_ = cv::Size2f(416, 416);
        dnn_ = cv::dnn::readNetFromONNX(modelPath + "416x416v.onnx");
        break;
    }
    }
}

std::vector<falpr::VehicleDetector::Vehicle> falpr::VehicleDetector::detect(const cv::Mat &image)
{
    if(image.empty())
        return std::vector<falpr::VehicleDetector::Vehicle>();

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1 / 255.f, modelSize_ , cv::Scalar(0, 0, 0), true, false, CV_32F);

    dnn_.setInput(blob);

    std::vector<cv::Mat> outputs;
    dnn_.forward(outputs, dnn_.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    bool yolov8 = false;
    if (dimensions > rows)
    {
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }

    float *data = (float *)outputs[0].data;

    float x_factor = (image.cols*1.f) / modelSize_.width;
    float y_factor = (image.rows*1.f) / modelSize_.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < rows; ++i)
    {
        if (yolov8)
        {
            float *classes_scores = data+4;

            cv::Mat scores(1, classes_.size(), CV_32FC1, classes_scores);
            cv::Point class_id;

            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > scoreThreshold_)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold_, nmsThreshold_, nms_result);

    std::vector<Vehicle> detections;
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Vehicle result;
        result.confidence = confidences[idx];
        result.type = static_cast<Type>(class_ids[idx]);
        result.boundingBox = boxes[idx];

        detections.push_back(result);
    }

    return detections;
}
