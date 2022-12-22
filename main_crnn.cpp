#include "Utils_crnn.h"
#include "Crnn.h"
#include <iostream>
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
    
    //这里模型路径使用绝对路径
    
    //不使用指针（int8）
//     CRNN_Recognize crnn_recognize("/src/notebooks/ncnn-20220420/build/tools/quantize/id_crnn_mobile_fix_sim_int8.param");
//     string rec_result=crnn_recognize.detect_crnn(m);
    
    
    ///使用指针身份证识别（int8）
//     CRNN_Recognize *cR=new CRNN_Recognize("/src/notebooks/ncnn-20220420/build/tools/quantize/id_crnn_mobile_fix_sim_int8.param");
//     string rec_result=cR->detect_crnn(m);
    
    
    
    ////fp16量化模型的推理
//      CRNN_Recognize *cR=new CRNN_Recognize("/src/notebooks/c++_ID_Img_Rec_Project/onnx2ncnn_model/crnn_sfz/id_crnn_mobile_fix_sim_fp16.param");
//     string rec_result=cR->detect_crnn(m); 
    
    //
    CRNN_Recognize *cR=new CRNN_Recognize("/src/notebooks/c++_ID_Img_Rec_Project/onnx2ncnn_model/crnn_sfz/id_crnn_mobile_fix_sim_int8.param");
    string rec_result=cR->detect_crnn(m);
    cout<<rec_result<<endl;
    
    return 0;
}