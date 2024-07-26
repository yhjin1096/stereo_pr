#ifndef PR_DBOW2_H
#define PR_DBOW2_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "DBoW2/DBoW2.h"

class PR_DBoW2
{
    private:
    public:
        PR_DBoW2(){};
        ~PR_DBoW2(){};

        OrbDatabase db;

        std::vector<std::vector<cv::Mat>> features;

        void stackFeatures(const std::vector<cv::Mat>& descriptors)
        {
            features.push_back(descriptors);
        }
        
        void createVocabulary()
        {
            // branching factor and depth levels 
            const int k = 9;
            const int L = 3;
            const DBoW2::WeightingType weight = DBoW2::TF_IDF;
            const DBoW2::ScoringType scoring = DBoW2::L1_NORM;

            OrbVocabulary voc(k, L, weight, scoring);

            std::cout << "Creating a small " << k << "^" << L << " vocabulary..." << std::endl;
            voc.create(features);
            std::cout << "... done!" << std::endl;

            std::cout << "Vocabulary information: " << std::endl
            << voc << std::endl << std::endl;

            // lets do something with this vocabulary
            std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;
            DBoW2::BowVector v1, v2;
            
            int NIMAGES = features.size();

            for(int i = 0; i < NIMAGES; i++)
            {
                voc.transform(features[i], v1);
                for(int j = 0; j < NIMAGES; j++)
                {
                    voc.transform(features[j], v2);
                    
                    // double score = voc.score(v1, v2);
                    // std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
                }
            }

            // save the vocabulary to disk
            std::cout << std::endl << "Saving vocabulary..." << std::endl;
            voc.save("kitti.yml.gz");
            std::cout << "Done" << std::endl;
        }

        void createDatabase(const std::string& voc_path)
        {
            std::cout << "Creating a database..." << std::endl;

            // load the vocabulary from disk
            OrbVocabulary voc;
            voc.loadFromTextFile(voc_path);
            
            OrbDatabase db(voc, false, 0);
            
            int nimages = features.size();
            for(int i = 0; i < nimages; i++)
            {
                db.add(features[i]);
            }

            std::cout << "Saving database..." << std::endl;
            db.save("dbow2_db/ORBdb.yml.gz");
            std::cout << "... done!" << std::endl;
        }

        void loadDatabase(const std::string path)
        {
            std::cout << "load Database..." << std::endl;
            db.load(path);
        }

        std::vector<int> queryDatabase(const int& node_idx, const std::vector<cv::Mat>& feature, int num)
        {
            DBoW2::QueryResults res;
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