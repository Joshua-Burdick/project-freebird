#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <ctime>

#define WINDOW_TITLE "Project Freebird"

std::string getCurrentDateTime();
void annotateImage(cv::VideoCapture&, cv::Mat&);

int main(int, char**)
{
    cv::Mat frame;
    cv::VideoCapture cap;

    // Open the default camera using default API
    cap.open(0);

    // Check if capture opened successfully
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    
    while (1) { // Run until terminated
        // Wait for a new frame from camera and store it into frame
        cap.read(frame);

        // Check if frame successfully grabbed
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        
        annotateImage(cap, frame);

        // Show live and wait for a key press
        cv::imshow(WINDOW_TITLE, frame);
        if (cv::waitKey(5) == 27)
            break;
    }
    
    cap.release();

    return 0;
}

std::string getCurrentDateTime() {
    time_t timeRaw;
    struct tm * timeInfo;
    char buffer[80];

    time (&timeRaw);
    timeInfo = localtime(&timeRaw);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeInfo);
    std::string str(buffer);

    return str;
}

void annotateImage(cv::VideoCapture& cap, cv::Mat& frame) {
    cv::putText(
        frame,
        "FPS: " + std::to_string(cap.get(cv::CAP_PROP_FPS)),
        cv::Point(10, 30),
        cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
        0.4,
        cv::Scalar(0, 255, 0)
    );
    cv::putText(
        frame,
        "Frame Size: " + std::to_string(cap.get(cv::CAP_PROP_FRAME_WIDTH)) + "x" + std::to_string(cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
        cv::Point(10, 50),
        cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
        0.4,
        cv::Scalar(0, 255, 0)
    );
    cv::putText(
        frame,
        "Time: " + getCurrentDateTime(),
        cv::Point(10, 70),
        cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
        0.4,
        cv::Scalar(0, 255, 0)
    );
}