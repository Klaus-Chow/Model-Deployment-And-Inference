#include "yolov5_virtual.h"
#include "Utils_yolo.h"
#include <iostream>
#include <Net_Detect.h>
using namespace std;
int main(int argc, char** argv)
{
    if(argc!=2)
    {
        fprintf(stderr,"Usage: %s [imagepath]\n",argv[0]);
        return -1;
    }
    const char* imagepath=argv[1];
    string img_name=imagepath;
    
    cv::Mat m=cv::imread(imagepath,1); //BGR
    if(m.empty())
    {
        fprintf(stderr,"cv::imread %s failed\n",imagepath);
        return -1;
    }
    std::vector<Object> objects;
//     YoloInfer yolo_detect("/src/notebooks/c++_ID_Img_Rec_Project/onnx2ncnn_model/yolo_mtz/best-mtz_sim_int8.param",true,0.25,0.45);
//     yolo_detect.detect(m,objects);
    
//     //画出bbox
//     draw_objects(m,objects,img_name);
    
//     //扣除ROI
//     cv::Mat ROI;
//     crop_ROI(m,ROI,objects,img_name);
    
   
    
    ////////////////测试一下Utils/Net_Detect
    YoloInfer *yolo_detect_p=new YoloInfer("/src/notebooks/c++_ID_Img_Rec_Project/onnx2ncnn_model/yolo_mtz/best-mtz_sim_int8.param",true,0.25,0.45);
    
    Net_detect(yolo_detect_p,m,objects);
    //画出bbox
    draw_objects(m,objects,img_name);
    
    //扣除ROI
    cv::Mat ROI;
    crop_ROI(m,ROI,objects,img_name);
    
        
        
    return 0;
}