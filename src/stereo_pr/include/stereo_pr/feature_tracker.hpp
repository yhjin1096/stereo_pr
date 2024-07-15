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
};

#endif