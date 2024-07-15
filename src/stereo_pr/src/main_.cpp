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
    Mode mode = Mode::query;

    for(int i = 0; i < num_images; i++)
    {
        // Node node(i, color_path, true);
        Node node(i, gray_path, false);
        ft.featureExtract(node.left_cam);
        ft.featureExtract(node.right_cam);

        node.left_cam.changeStructure();
        node.right_cam.changeStructure();

        // cv::Mat draw_image = node.left_cam.image.clone();
        // cv::drawKeypoints(node.left_cam.image, node.left_cam.keypts, draw_image);
        // cv::imshow("draw_image", draw_image);
        // char k = cv::waitKey(1);
        // if(k == 'q')
        //     break;
        
        if(mode == Mode::save)
            pr.stackFeatures(node.left_cam.row_descriptor);
        else if(mode == Mode::query)
        {
            if(i == 0)
                pr.loadDatabase("ORBdb.yml.gz");
            
            pr.queryDatabase(node.index, node.left_cam.row_descriptor);
        }
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