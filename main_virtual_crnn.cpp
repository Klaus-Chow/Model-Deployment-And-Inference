#include "Crnn2.h"
using namespace std;
int main(int argc, char** argv)
{
    //单独测试
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
    const char* img_path = argv[1];
    cv::Mat m = cv::imread(img_path, 1); //BGR
//         cv::imwrite("../Output/ROI_sfz_ori.jpg",m);
    if (m.empty())
    {
         fprintf(stderr, "cv::imread %s failed\n", img_path);
         return -1;
    }
    
    CRNN_Recognize *cR=new CRNN_Recognize("/src/notebooks/c++_ID_Img_Rec_Project/onnx2ncnn_model/crnn_sfz/id_crnn_mobile_fix_sim_int8.param",true);
    string rec;
    cR->detect(m,rec);
    
    cout<<rec<<endl;
    return 0;

}