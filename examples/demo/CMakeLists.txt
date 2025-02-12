cmake_minimum_required(VERSION 3.10)
project(demo)

# Указываем стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Находим OpenCV
find_package(OpenCV REQUIRED)
find_package(falpr REQUIRED)

# Создаем исполняемый файл
add_executable(${PROJECT_NAME} main.cpp)

# Линкуем с библиотеками OpenCV и FALPR
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} falpr)
