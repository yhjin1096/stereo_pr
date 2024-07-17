#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

const std::string imageExtensions[] = {".jpg", ".jpeg", ".png", ".gif", ".bmp"};

class Timer
{
public:
    int aver_time;
    std::chrono::system_clock::time_point start_time;
    void tic(void)
    {
        start_time = std::chrono::system_clock::now();
    }
    void toc(void)
    {
        std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
        std::chrono::milliseconds mill = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "ms = " << mill.count() << std::endl;
        std::cout << "s = " << mill.count()/1000.0 << std::endl;

        if(aver_time==0){aver_time+=static_cast<int>(mill.count());}
        else
        {
            aver_time+=mill.count();
            aver_time/=2;
            std::cout << "aver ms = " << aver_time << std::endl;
        }
    }
    Timer():aver_time(0){}
    ~Timer()=default;
};

class utils
{
    private:
    public:
        static int CountImages(const std::string &path)
        {
            int num_images = 0;
            try
            {
                // 지정된 폴더 내의 모든 파일에 대해 반복
                for (const auto &entry : boost::filesystem::directory_iterator(path))
                {
                    // 디렉토리인 경우 건너뛰기
                    if (boost::filesystem::is_directory(entry.path()))
                        continue;

                    // 이미지 파일인 경우 개수 증가
                    for (const std::string &ext : imageExtensions)
                    {
                        if (entry.path().extension() == ext)
                            num_images++;
                    }
                }

                // std::cout << "Number of image files in the folder: " << num_images << std::endl;
            }
            catch (const std::exception &ex)
            {
                std::cerr << "Error: " << ex.what() << std::endl;
                return 0;
            }
            return num_images;
        }

        static std::vector<cv::Mat> loadParameters(const std::string& filename) {
            std::vector<cv::Mat> parameters;
            std::ifstream file(filename);
            std::string line;
            
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                std::string header;
                iss >> header; // P0, P1, etc.

                cv::Mat param(3, 4, CV_64F);
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        iss >> param.at<double>(i, j);
                    }
                }
                parameters.emplace_back(param);
            }
            
            return parameters;
        }

        static cv::Point2i getPixelPosition(const cv::Mat& pose, cv::Mat& traj)
        {
            cv::Point2i point;
            point.x = traj.size().width/2 + pose.at<double>(0,3);
            point.y = traj.size().height/2 + pose.at<double>(2,3);
            return point;
        }
};
#endif