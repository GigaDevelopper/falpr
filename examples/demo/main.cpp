#include <opencv2/opencv.hpp>
#include "FALPR/FALPR.h"

int main(int argc, char** argv) {
  std::string image_path = "/home/azmiddin/my/falpr/dataset/vehicles/uz/50Z900AA_Car.jpg";
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    //return -1;
  }

  // Загружаем изображение
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "Error: Could not load image " << argv[1] << std::endl;
    return -1;
  }

  // Создаем объект FALPR с параметрами
  falpr::FALPR recognizer(0.7f, 0.85f, falpr::PlateDetector::ModelSize::Large, "/home/azmiddin/my/falpr/models/");

  // Выполняем распознавание
  auto results = recognizer.recognize(image);

  // Отображаем результат
  for (const auto& result : results) {
    recognizer.drawResult(image, result, cv::Scalar(0, 255, 0));
  }

  // Показываем изображение
  cv::imshow("FALPR Result", image);
  cv::waitKey(0);

  return 0;
}
