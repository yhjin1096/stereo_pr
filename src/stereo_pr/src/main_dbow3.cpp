#include "stereo_pr/stereo_pr_dbow3.hpp"

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
    PR_DBoW3 pr;
    Visualizer viz;
    Mode mode = Mode::query;

    std::vector<Node> nodes;
    cv::Mat traj = cv::Mat::zeros(cv::Size(1000,1000), CV_8UC3);
    int wait_sec = 1;
    Timer timer;

    for(int i = 0; i < num_images; i++)
    {
        // Node node(i, color_path, true);
        Node node(i, gray_path, false);
        ft.featureExtract(node.left_cam);
        ft.featureExtract(node.right_cam);

        if(i != 0 && mode != Mode::save)
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
            ft.calcPose(nodes[i-1], tmp_node);
            node.cam_to_world_pose = tmp_node.cam_to_world_pose.clone();
            node.world_to_cam_pose = tmp_node.world_to_cam_pose.clone();
            node.rot_rodrigues = tmp_node.rot_rodrigues.clone();
            node.translation = tmp_node.translation.clone();
            viz.vizTrajectory(node.cam_to_world_pose, traj);
        }

        cv::Mat pr_traj = traj.clone();
        if(mode == Mode::save)
            pr.stackFeatures(node.left_cam.descriptors);
        else
        {
            if(i == 0)
            {
                timer.tic();
                pr.loadDatabase("dbow3_db/ORBdb.yml.gz");
                timer.toc();
            }
            else
            {
                // vo // node[i]
                std::vector<int> pr_list = pr.queryDatabase(node.index, node.left_cam.descriptors, 10);
                if(pr_list.size())
                {
                    viz.vizPRTrajectory(nodes, node, pr_list, pr_traj);
                    wait_sec = 0;
                }
                else
                    wait_sec = 1;
            }
        }

        nodes.push_back(node);
        char key = cv::waitKey(wait_sec);
        if(key == 27)
            break;
    }

    if(mode == Mode::save) // database 생성
    {
        // pr.VocCreation();
        pr.createDatabase("ORBvoc.txt");
    }

    return 0;
}