#include "stereo_pr/stereo_pr_.hpp"

enum Mode
{
    save,
    query
};

int main(int argc, char **argv)
{
    std::string gray_path = "/home/cona/Downloads/dataset/data_odometry_gray/dataset/sequences/00/";
    std::string color_path = "/home/cona/Downloads/dataset/data_odometry_color/dataset/sequences/00/";
    int num_images = utils::CountImages(gray_path + "image_0/");
    std::cout << "num_images: " << num_images << std::endl;

    FeatureTracker ft;
    PlaceRecognition pr;
    Visualizer viz;
    Mode mode = Mode::query;

    std::vector<Node> nodes;
    cv::Mat traj = cv::Mat::zeros(cv::Size(1000,1000), CV_8UC3);

    for(int i = 0; i < num_images; i++)
    {
        // Node node(i, color_path, true);
        Node node(i, gray_path, false);
        ft.featureExtract(node.left_cam);
        ft.featureExtract(node.right_cam);

        node.left_cam.changeStructure();
        node.right_cam.changeStructure();

        if(i != 0)
        {
            // get stereo matching
            nodes[i-1].stereo_matches = ft.getDescMatching(nodes[i-1].left_cam.descriptors, nodes[i-1].right_cam.descriptors);
            viz.vizMatches(nodes[i-1], true);
            nodes[i-1].removeMatchingOutlier();

            // tracking
            Node tmp_node = ft.trackImage(nodes[i-1], node);
            viz.vizTracking(nodes[i-1], tmp_node, true);

            // get 3d point
            nodes[i-1].points3D = ft.calc3DPoints(nodes[i-1].left_cam, nodes[i-1].right_cam);

            // pose estimation
            // nodes[i-1], tmp_node
            // node의 pose를 tmp_node의 pose로 
            ft.calcPose(nodes[i-1], tmp_node);
            node.cam_to_world_pose = tmp_node.cam_to_world_pose.clone();
            node.world_to_cam_pose = tmp_node.world_to_cam_pose.clone();
            node.rot_rodrigues = tmp_node.rot_rodrigues.clone();
            node.translation = tmp_node.translation.clone();
            viz.vizTrajectory(node.cam_to_world_pose, traj);

            // if(mode == Mode::save)
            //     pr.stackFeatures(node.left_cam.row_descriptor);
            // else if(mode == Mode::query)
            // {
            //     if(i == 0)
            //         pr.loadDatabase("ORBdb.yml.gz");
            //     else
            //     {
            //         // vo // node[i]
            //         int pr_node_idx = pr.queryDatabase(node.index, node.left_cam.row_descriptor);
            //     }
            // }
        }

        nodes.push_back(node);
        char key = cv::waitKey(1);
        if(key == 27)
            break;
    }
    
    Timer timer;
    timer.tic();
    if(mode == Mode::save) // database 생성
    {
        // pr.VocCreation();
        pr.createDatabase("ORBvoc.txt");
    }
    timer.toc();
    
    return 0;
}