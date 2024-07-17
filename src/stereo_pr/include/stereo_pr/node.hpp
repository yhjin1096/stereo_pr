#ifndef NODE_HPP
#define NODE_HPP

#include <math.h>
#include "stereo_pr/camera.hpp"

class Node
{
    private:
        
    public:
        int index;
        Camera left_cam, right_cam;
        double base_line;

        cv::Mat rot_rodrigues = cv::Mat::zeros(3, 1, CV_64F), translation = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat world_to_cam_pose = cv::Mat::eye(4, 4, CV_64F); //cam 기준 world
        cv::Mat cam_to_world_pose = cv::Mat::eye(4, 4, CV_64F); //world 기준 cam
        // cv::Mat world_pose = cv::Mat::eye(4, 4, CV_64F);

        std::vector<cv::DMatch> stereo_matches;
        cv::Mat points3D;

        Node(const int& idx, const std::string& path, bool is_color)
        {
            index = idx;
            // 0 - gray left, 1 - gray right, 2 - color left, 3 - color right
            if(!is_color)
            {
                Camera tmp_left(0, idx, path);
                Camera tmp_right(1, idx, path);
                left_cam = tmp_left.getOwn();
                right_cam = tmp_right.getOwn();
            }
            else
            {
                Camera tmp_left(2, idx, path);
                Camera tmp_right(3, idx, path);
                left_cam = tmp_left.getOwn();
                right_cam = tmp_right.getOwn();
            }

            base_line = calcBaseline();

            // cam_to_world_pose.at<double>(0,3) = base_line;
            world_to_cam_pose = cam_to_world_pose.inv();  
        };
        ~Node(){};

        void removeMatchingOutlier()
        {
            std::vector<cv::Point2f> left_point2f, right_point2f;
            std::vector<cv::KeyPoint> left_keypoint, right_keypoint;
            
            for(int i = 0; i < stereo_matches.size(); i++)
            {
                int query_idx = stereo_matches[i].queryIdx;//left
                int train_idx = stereo_matches[i].trainIdx;//right
                
                left_keypoint.emplace_back(left_cam.keypts[query_idx]);
                right_keypoint.emplace_back(right_cam.keypts[train_idx]);
                left_point2f.emplace_back(left_cam.keypts[query_idx].pt);
                right_point2f.emplace_back(right_cam.keypts[train_idx].pt);
            }

            left_cam.keypts.swap(left_keypoint);
            right_cam.keypts.swap(right_keypoint);
            left_cam.keypoints.swap(left_point2f);
            right_cam.keypoints.swap(right_point2f);
        }

        double calcBaseline()
        {
            cv::Mat tmp1 = left_cam.projection_mat.clone();
            cv::Mat tmp2 = right_cam.projection_mat.clone();
            cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
            cv::vconcat(tmp1, addup, tmp1);
            cv::vconcat(tmp2, addup, tmp2);

            cv::Mat result_mat = tmp2.inv() * tmp1;
            double result = result_mat.at<double>(0,3);

            return result;
        }
};

#endif