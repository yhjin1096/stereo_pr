#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "stereo_pr/node.hpp"

class Visualizer
{
    private:

    public:
        void vizMatches(const Node& node, bool viz)
        {
            if(viz)
            {
                cv::Mat output;
                cv::drawMatches(node.left_cam.image, node.left_cam.keypts, 
                                node.right_cam.image, node.right_cam.keypts,
                                node.stereo_matches, output);
                cv::imshow("stereo matches", output);
            }
        }

        void vizTracking(const Node& refer, const Node query, bool viz)
        {
            if(viz)
            {
                cv::Mat output = refer.left_cam.image.clone();
                for(int i = 0; i < refer.left_cam.keypoints.size(); i++)
                {
                    cv::circle(output, refer.left_cam.keypoints[i], 2, CV_RGB(0,255,0));
                }
                for(int i = 0; i < query.left_cam.keypoints.size(); i++)
                {
                    cv::circle(output, query.left_cam.keypoints[i], 2, CV_RGB(0,0,255));
                }
                for(int i = 0; i < refer.left_cam.keypoints.size(); i++)
                {
                    cv::line(output, refer.left_cam.keypoints[i], query.left_cam.keypoints[i], CV_RGB(0,255,0));
                }
                // cv::resize(output,output,output.size()/2);
                cv::imshow("tracking", output);
            }
        }

        void vizTrajectory(const cv::Mat& pose, cv::Mat& traj)
        {
            cv::Point2i point;
            point.x = traj.size().width/2 + pose.at<double>(0,3);
            point.y = traj.size().height/2 + pose.at<double>(2,3);
            cv::circle(traj, point, 1, cv::Scalar(0,0,255), 1);
            cv::imshow("traj", traj);
        }
};

#endif