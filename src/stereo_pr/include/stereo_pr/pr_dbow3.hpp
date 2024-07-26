#ifndef PR_DBOW3_H
#define PR_DBOW3_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "DBoW3/DBoW3.h"
#include "DBoW3/DescManip.h"

class PR_DBoW3
{
    private:
    public:
        DBoW3::Database db;

        std::vector<cv::Mat> features;

        void stackFeatures(const cv::Mat& descriptors)
        {
            features.push_back(descriptors);
        }

        void createDatabase(const std::string& voc_path)
        {
            std::cout << "Creating a database..." << std::endl;

            // load the vocabulary from disk
            DBoW3::Vocabulary voc(voc_path);
            // voc.loadFromTextFile(voc_path);
            
            DBoW3::Database db(voc, false, 0);
            
            int nimages = features.size();
            for(int i = 0; i < nimages; i++)
            {
                db.add(features[i]);
            }

            std::cout << "Saving database..." << std::endl;
            db.save("ORBdb.yml.gz");
            std::cout << "... done!" << std::endl;
        }

        void loadDatabase(const std::string path)
        {
            std::cout << "load Database..." << std::endl;
            db.load(path);
        }

        std::vector<int> queryDatabase(const int& node_idx, const cv::Mat& feature, int num)
        {
            DBoW3::QueryResults res;
            db.query(feature, res, num);
            // std::cout << "Image: " << node_idx << ", " << res << std::endl;
            
            std::vector<int> id_list;

            for(int i = 0; i < res.size(); i++)
            {
                int res_idx = static_cast<int>(res[i].Id);
                if(node_idx - res_idx >= 20 && res[i].Score >= 0.05) // 현재 노드(node_idx)보다 이전 노드이고 20노드 이상 차이날 때
                    id_list.push_back(res_idx);
                    // return res_idx;
            }

            return id_list;
        }
};

#endif