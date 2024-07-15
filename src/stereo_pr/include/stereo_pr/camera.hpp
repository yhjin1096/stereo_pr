#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "stereo_pr/utils.hpp"

class Camera
{
    private:
        
    public:
        cv::Mat projection_mat;
        cv::Mat intrinsic_mat;

        cv::Mat image, gray_image;
        cv::Mat descriptors;
        std::vector<cv::Mat> row_descriptor;
        std::vector<cv::KeyPoint> keypts;
        std::vector<cv::Point2f> keypoints;

        Camera(){};
        Camera(const int& cam_idx, const int& image_idx, const std::string& file_path)
        {
            image = cv::imread(file_path + "image_" + std::to_string(cam_idx) + cv::format("/%06d.png", image_idx));
            gray_image = cv::imread(file_path + "image_" + std::to_string(cam_idx) + cv::format("/%06d.png", image_idx), 0);

            std::vector<cv::Mat> params = utils::loadParameters(file_path + "calib.txt");
            projection_mat = params[cam_idx].clone();
            intrinsic_mat = projection_mat(cv::Rect(0,0,3,3)).clone();
            // cv::imshow("image",image);
            // cv::waitKey(0);
        }
        ~Camera(){};

        Camera& getOwn(){return *this;};

        void changeStructure()
        {
            row_descriptor.resize(descriptors.rows);

            for(int i = 0; i < descriptors.rows; ++i)
            {
                row_descriptor[i] = descriptors.row(i);
            }
        }
};

#endif