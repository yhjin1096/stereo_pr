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

                cv::resize(output, output, output.size()/3*2);
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
                cv::resize(output, output, output.size()/3*2);
                cv::imshow("tracking", output);
            }
        }

        void vizTrajectory(const cv::Mat& pose, cv::Mat& traj)
        {
            cv::Point2i point = utils::getPixelPosition(pose, traj);
            cv::circle(traj, point, 1, cv::Scalar(255,255,255), 1);
            cv::imshow("traj", traj);
        }

        void vizPRTrajectory(const std::vector<Node>& nodes, const Node& curr_node, const std::vector<DBoW2::Result>& pr_list, cv::Mat& image)
        {
            cv::circle(image, utils::getPixelPosition(curr_node.cam_to_world_pose, image), 4, CV_RGB(255, 0, 0));
            for(int i = 0; i < pr_list.size(); i++)
            {
                int pr_id = pr_list[i].Id;
                cv::Mat nodes_pose = nodes[pr_id].cam_to_world_pose;
                cv::Mat curr_node_pose = curr_node.cam_to_world_pose;
                // std::cout << curr_node.index << "," << pr_id << ": " << pr_list[i].Score << std::endl;

                if(i == 0)
                {
                    cv::circle(image, utils::getPixelPosition(nodes_pose, image), 5, CV_RGB(0, 255, 0), 3);
                    cv::line(image, utils::getPixelPosition(curr_node_pose, image),
                                    utils::getPixelPosition(nodes_pose, image), CV_RGB(0,255,0), 3);
                }
                else
                {
                    cv::circle(image, utils::getPixelPosition(nodes_pose, image), 5, CV_RGB(255, 255, 0), 3);
                    cv::line(image, utils::getPixelPosition(curr_node_pose, image),
                                    utils::getPixelPosition(nodes_pose, image), CV_RGB(255,255,0), 1);
                }
            }
            cv::imshow("pr_query", image);
        }
};

#endif