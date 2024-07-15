#include "stereo_pr/stereo_pr.hpp"

int main(int argc, char **argv)
{
    std::string left_path = "/home/cona/Downloads/dataset/data_odometry_gray/dataset/sequences/00/image_0/";
    std::string right_path = "/home/cona/Downloads/dataset/data_odometry_gray/dataset/sequences/00/image_1/";
    std::vector<cv::Mat> gt_poses;
    int num_images = 0;
    CountImages(num_images, left_path);
    readGTPose("/home/cona/Downloads/dataset/data_odometry_gray/data_odometry_poses/dataset/poses/00.txt", gt_poses);
    

    Tracker tracker;
    Visualizer visualizer;
    int gap = 1;
    
    cv::Mat traj = cv::Mat::zeros(cv::Size(1000,1000), CV_8UC3);
    cv::Mat prev_pose = cv::Mat::eye(4, 4, CV_64F);
    
    for(int i = 0; i < num_images-gap; i++)
    {
        Node refer, query;

        refer.left_cam.loadImage(left_path + cv::format("%06d.png", i));
        refer.right_cam.loadImage(right_path + cv::format("%06d.png", i));
        query.left_cam.loadImage(left_path + cv::format("%06d.png", i+gap));
        query.right_cam.loadImage(right_path + cv::format("%06d.png", i+gap));
        
        /*feature extract*/
        // tracker.extractKeypointsFAST(refer.left_cam); //FAST
        tracker.extractKeypointsORB(refer.left_cam); //ORB
        tracker.extractKeypointsORB(refer.right_cam); //ORB
        
        /*tracking*/
        /*trackKLTAndDescriptor: refer 좌우측에서 feature 추출*/
        tracker.trackKLTAndDescriptor(refer, query);
        
        /*get 3d point*/
        tracker.calc3DPoints(refer);

        /*calculate pose*/
        tracker.calcPose(refer, query);
        
        // std::cout << "frame" << i << " -> frame" << i+1 << std::endl; 
        // std::cout << query.left_cam.cam_to_world_pose << std::endl;
        // std::cout << "" << std::endl;
        
        prev_pose = prev_pose * query.left_cam.cam_to_world_pose;
        
        // cv::Mat viz_track, viz_extract;
        // visualizer.visualizeExtractAll(refer, query, viz_extract);
        // visualizer.visualizeTracking(refer, query, viz_track);
        // visualizer.visualizeRelative3D(myWindow, refer, query, gt_poses[i], gt_poses[i+1], i);
        visualizer.visualizeTrajectory(prev_pose, traj);
        char key = cv::waitKey(1);
        if(key == 'q')
            break;
    }
    
    return 0;
}