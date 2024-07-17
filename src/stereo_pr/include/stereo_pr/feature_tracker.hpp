#ifndef FEATURE_TRACKER_H
#define FEATURE_TRACKER_H

#include "stereo_pr/camera.hpp"

#include <opencv2/opencv.hpp>
// #include "opencv2/video/tracking.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
// #include "opencv2/calib3d/calib3d.hpp"

class FeatureTracker
{
    private:
    public:
        void featureExtract(Camera& cam)
        {
            // ORB 특징 검출기 생성
            cv::Ptr<cv::Feature2D> orb = cv::ORB::create(1500);

            // 특징 검출
            orb->detectAndCompute(cam.gray_image, cv::noArray(), cam.keypts, cam.descriptors);
            cv::KeyPoint::convert(cam.keypts, cam.keypoints, std::vector<int>());
        }

        std::vector<cv::DMatch> getDescMatching(const cv::Mat& query_desc, const cv::Mat& train_desc)
        {
            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
            std::vector<cv::DMatch> match, good_matches;
            matcher->match(query_desc, train_desc, match);
            
            double min_dist = 10000, max_dist = 0;
            for (int i = 0; i < query_desc.rows; i++) {
                double dist = match[i].distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
            }
            for (int i = 0; i < query_desc.rows; i++) {
                if (match[i].distance <= std::max(2 * min_dist, 25.0)) {
                    good_matches.emplace_back(match[i]);
                }
            }
            
            // cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, true);
            // std::vector<cv::DMatch> matches;
            // matcher->match(query_desc, train_desc, matches);
            
            // const int match_size = 150;//matches.size() * 0.5;
            // std::sort(matches.begin(), matches.end());
            // std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + match_size);

            return good_matches;
        }

        Node trackImage(Node& refer, Node& query)
        {
            Node res_node = query;
            std::vector<float> err;
            cv::Size winSize=cv::Size(21,21);
            cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
            std::vector<uchar> status0;

            calcOpticalFlowPyrLK(refer.left_cam.gray_image, res_node.left_cam.gray_image,
                                refer.left_cam.keypoints, res_node.left_cam.keypoints,
                                status0, err, winSize, 4, termcrit, 0, 0.001);
            
            int j = 0;
            for(int i = 0; i < status0.size(); i++)
            {
                if(!status0[i])
                {
                    refer.left_cam.keypts.erase(refer.left_cam.keypts.begin() + (i-j));
                    refer.right_cam.keypts.erase(refer.right_cam.keypts.begin() + (i-j));
                    refer.left_cam.keypoints.erase(refer.left_cam.keypoints.begin() + (i-j));
                    refer.right_cam.keypoints.erase(refer.right_cam.keypoints.begin() + (i-j));
                    res_node.left_cam.keypoints.erase(res_node.left_cam.keypoints.begin() + (i-j));
                    j++;
                }
            }

            return res_node;
        }

        cv::Mat calc3DPoints(const Camera& left_cam, const Camera& right_cam)
        {
            cv::Mat points4D, points3D;
            cv::triangulatePoints(left_cam.projection_mat,  right_cam.projection_mat,
                                  left_cam.keypoints, right_cam.keypoints, points4D);
            cv::convertPointsFromHomogeneous(points4D.t(), points3D);

            return points3D;
        }

        void calcPose(Node& refer, Node& query)
        {
            cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);   
            cv::Mat rvec       = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat inliers;

            int iterationsCount = 500;        // number of Ransac iterations.
            float reprojectionError = 0.1;    // maximum allowed distance to consider it an inlier.
            float confidence = 0.999;         // RANSAC successful confidence.
            bool useExtrinsicGuess = true;
            int flags =cv::SOLVEPNP_ITERATIVE;

            // rotation & translation => world to cam pose
            cv::solvePnPRansac(refer.points3D, query.left_cam.keypoints, refer.left_cam.intrinsic_mat, distCoeffs, rvec, translation,
                               useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                               inliers, flags );
            cv::Rodrigues(rvec, rotation);

            cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
            cv::Mat rigid_body_transformation; //relative pose
            cv::Mat frame_pose = refer.cam_to_world_pose.clone();
            
            if(abs(rotation_euler[1])<0.1 && abs(rotation_euler[0])<0.1 && abs(rotation_euler[2])<0.1)
            {
                integrateOdometry(0 , rigid_body_transformation, frame_pose, rotation, translation);

                query.cam_to_world_pose = frame_pose.clone();
                query.world_to_cam_pose = query.cam_to_world_pose.inv();
                query.rot_rodrigues = rvec;
                query.translation = translation;

            } else
            {
                query.cam_to_world_pose = refer.cam_to_world_pose;
                query.world_to_cam_pose = refer.world_to_cam_pose;
                query.rot_rodrigues = refer.rot_rodrigues;
                query.translation = refer.translation;
                std::cout << "Too large rotation"  << std::endl;
            }
        }

        void integrateOdometry(int frame_i, cv::Mat& rigid_body_transformation, cv::Mat& frame_pose, const cv::Mat& rotation, const cv::Mat& translation_stereo)
        {
            cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

            cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
            cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

            double scale = sqrt((translation_stereo.at<double>(0))*(translation_stereo.at<double>(0)) 
                                + (translation_stereo.at<double>(1))*(translation_stereo.at<double>(1))
                                + (translation_stereo.at<double>(2))*(translation_stereo.at<double>(2))) ;

            rigid_body_transformation = rigid_body_transformation.inv();
            
            // if ((scale>0.1)&&(translation_stereo.at<double>(2) > translation_stereo.at<double>(0)) && (translation_stereo.at<double>(2) > translation_stereo.at<double>(1))) 
            if (scale > 0.05 && scale < 10) 
            {

            frame_pose = frame_pose * rigid_body_transformation;

            }
            else 
            {
                frame_pose = frame_pose * rigid_body_transformation;
            std::cout << "[WARNING] scale is very low or very high: " << scale << std::endl;
            }
        }

        cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
        {
        
            assert(isRotationMatrix(R));
            
            float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
        
            bool singular = sy < 1e-6; // If
        
            float x, y, z;
            if (!singular)
            {
                x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
                y = atan2(-R.at<double>(2,0), sy);
                z = atan2(R.at<double>(1,0), R.at<double>(0,0));
            }
            else
            {
                x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
                y = atan2(-R.at<double>(2,0), sy);
                z = 0;
            }
            return cv::Vec3f(x, y, z);
            
        }

        bool isRotationMatrix(cv::Mat &R)
        {
            cv::Mat Rt;
            transpose(R, Rt);
            cv::Mat shouldBeIdentity = Rt * R;
            cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
            
            return  norm(I, shouldBeIdentity) < 1e-6;
            
        }
};

#endif