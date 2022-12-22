#include "Yolov5.h"
#include "Utils_yolo.h"
#include<iostream>
#include<string>
using namespace std;
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
    const char* imagepath = argv[1];
    string img_name=imagepath;
    
    cv::Mat m = cv::imread(imagepath, 1); //BGR
//     cout<<m.cols<<" "<<m.rows<<endl;//w,h
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    std::vector<Object> objects;
    
    //身份证检测"/src/notebooks/ncnn-20220420/build/tools/quantize/best_lite-sim_int8.param"  pro_threshold:0.5,输出node:output、471、491
    //门头照检测"./onnx2ncnn_model/best-mtz_sim_fp16.param" pro_threshold:0.25 输出node:output、417、437
    
    yoloInfer yolo_detect("/src/notebooks/c++_ID_Img_Rec_Project/onnx2ncnn_model/yolo_mtz/best-mtz_sim_int8.param",0.25);
    yolo_detect.detect(m,objects);
    //画出bbox
    draw_objects(m, objects,img_name);
    //扣除ROI
    cv::Mat ROI;
    crop_ROI(m,ROI,objects,img_name);

    
    return 0;
}