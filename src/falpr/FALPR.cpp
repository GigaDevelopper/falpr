//
// Created by azmiddin on 2/12/25.
//
#include "FALPR/FALPR.h"
#include "FALPR/Utils.h"

#include <iomanip>
#include <iostream>
#include <opencv2/imgproc.hpp>

falpr::FALPR::FALPR(const float &charConfidence,
                    const float &overallConfidence,
                    const PlateDetector::ModelSize &modelSize,
                    const std::string &modelPath):
    plateDetector_(modelSize, modelPath),
    plateRecognizer_(charConfidence, modelPath),
    overallConfidence_(overallConfidence),
    vehicleDetector_(modelPath, static_cast<VehicleDetector::ModelSize>(modelSize))
{
}

std::vector<falpr::FALPR::Result> falpr::FALPR::recognize(const cv::Mat &image)
{
    std::vector<Result> results;

    // detect vehicles
    std::vector<VehicleDetector::Vehicle> detectedVehicles = vehicleDetector_.detect(image);
    if(detectedVehicles.empty())
        detectedVehicles.push_back(VehicleDetector::Vehicle{VehicleDetector::Type::CAR,
                                                            100,
                                                            cv::Rect(0, 0, image.cols, image.rows)});

    for(const VehicleDetector::Vehicle &vehicle: detectedVehicles) {
        // detect license plate
        cv::Mat croppedVehicle = falpr::utils::cropFrame(image, vehicle.boundingBox);
        for(PlateDetector::Plate &plate: plateDetector_.detect(croppedVehicle)) {
            // recognize license plate
            cv::Mat croppedPlate = PlateDetector::cropPlate(croppedVehicle, plate);
            PlateRecognizer::License license = plateRecognizer_.recognize(croppedPlate);

            //rescale plate keypoints to original image
            for(cv::Point2f &kp: plate.keypoints) {
                kp.x += vehicle.boundingBox.x;
                kp.y += vehicle.boundingBox.y;
            }

            // rescale license keypoints to original image
            for(PlateRecognizer::Char &c: license.characters) {
                c.boundingBox.x += plate.keypoints[0].x;
                c.boundingBox.y += plate.keypoints[0].y;
            }

            // calculate overall confidence
            if(vehicle.confidence*0.05 + plate.confidence*0.05 + license.totalConfidence*0.9 >= overallConfidence_)
                results.push_back(Result{vehicle,
                                         plate,
                                         license});
        }
    }

    // if no license plate found just return the first vehicle itself
    if(results.empty())
        results.push_back(Result{detectedVehicles.front(),
                                 PlateDetector::Plate{},
                                 PlateRecognizer::License{}});

    return results;
}

void falpr::FALPR::drawResult(cv::Mat &image,
                              const Result &result,
                              const cv::Scalar &color)
{
    cv::rectangle(image, result.vehicle.boundingBox, color, 2);

    // draw result
    // Calculate text position (center-bottom of bounding box)
    cv::Point textPosition(result.vehicle.boundingBox.x + result.vehicle.boundingBox.width / 2,
                           result.vehicle.boundingBox.y + result.vehicle.boundingBox.height + 20);

    // Draw text at textPosition
    switch(result.vehicle.type)
    {
    case VehicleDetector::BUS: {
        cv::putText(image, "Bus", textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        break;
    }
    case VehicleDetector::CAR: {
        cv::putText(image, "Car", textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        break;
    }
    case VehicleDetector::MOTOR: {
        cv::putText(image, "Motor", textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        break;
    }
    case VehicleDetector::TRUCK: {
        cv::putText(image, "Truck", textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        break;
    }
    case VehicleDetector::VAN: {
        cv::putText(image, "Van", textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        break;
    }
    default: {
        cv::putText(image, "Unknown", textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        break;
    }
    }

    // draw license plate
    if(result.license.characters.empty())
        return;

    // get confidence string with precision
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << result.license.totalConfidence*100;

    std::string infoString = "LP: " + result.license.license + " Conf: " + stream.str() + "%";
    cv::Size textSize = cv::getTextSize(infoString, cv::FONT_HERSHEY_DUPLEX, 1, 1, nullptr);

    // FIXME: textbox may be out of image region
    cv::Rect textBox(result.plate.keypoints[0].x - 20, result.plate.keypoints[0].y - 50, textSize.width + 40, textSize.height + 30);

    cv::rectangle(image, textBox, cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(image, infoString, cv::Point(result.plate.keypoints[0].x + 5, result.plate.keypoints[0].y - 10), cv::FONT_HERSHEY_DUPLEX, 1, color, 1, 0);

    for (auto &kp: result.plate.keypoints)
        cv::circle(image, kp, 6, cv::Scalar(rand()%256, rand()%256, rand()%256), cv::FILLED);
}
