#include "FALPR/PlateRecognizer.h"

#include "FALPR/Utils.h"

falpr::PlateRecognizer::PlateRecognizer(const float &threshold, const std::string &modelPath) :
    scoreThreshold_(threshold),
    modelSize_(cv::Size2f(576, 128))
{
    dnn_ = cv::dnn::readNetFromONNX(modelPath + "ocr_uz.onnx");
}

falpr::PlateRecognizer::License falpr::PlateRecognizer::recognize(const cv::Mat &image)
{
    if(image.empty())
        return License{};

    // detect
    std::vector<Char> chars = runInference(image);
    std::sort(chars.begin(), chars.end());

    // get total confidence
    float totalConfidence = 0;
    std::string license = "";
    for(const auto &c: chars){
        totalConfidence += c.confidence;
        license += c.label;
    }

    // postprocess - pattern validation
    if(!falpr::utils::isValidUz(license))
        return License{};

    return License{Uz, totalConfidence/chars.size(), license, chars};
}

std::vector<falpr::PlateRecognizer::Char> falpr::PlateRecognizer::runInference(const cv::Mat &input)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1.0/255.0, modelSize_, cv::Scalar(), true, false);

    dnn_.setInput(blob);

    std::vector<cv::Mat> outputs;
    dnn_.forward(outputs, dnn_.getUnconnectedOutLayersNames());

    const int rows = outputs[0].size[2];
    const int dimensions = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    float *data = (float *)outputs[0].data;

    float x_factor = (input.cols*1.f) / modelSize_.width;
    float y_factor = (input.rows*1.f) / modelSize_.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < rows; ++i)
    {
        float *classes_scores = data+4;
        cv::Mat scores(1, classes_.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

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
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold_, nmsThreshold_, nms_result);

    std::vector<Char> detections;
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Char result;
        result.label = std::toupper(classes_[class_ids[idx]]);
        result.confidence = confidences[idx];
        result.boundingBox = boxes[idx];

        detections.push_back(result);
    }

    return detections;
}
