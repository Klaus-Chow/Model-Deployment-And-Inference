#ifndef YOLOV5_H
#define YOLOV5_H
#include<iostream>
#include<string>
#include "layer.h"
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
using namespace std;
struct Object
{
    cv::Rect_<float>rect;
    int label;
    float prob;
};
class yoloInfer
{
    public: 
        yoloInfer(string path,float pro_threshold):model_path(path),prob_threshold(pro_threshold)
        {
            cout<<"yolo初始化"<<endl;
            string model_param_path=model_path;
            string model_bin_path=model_path.replace(model_param_path.rfind("."),6,".bin");
            
            const char *param_path=model_param_path.c_str();
            const char *bin_path=model_bin_path.c_str();
    
            cout<<" "<<bin_path<<endl;
            
            //使用量化8推理模式
            yolov5.opt.num_threads=8;
            yolov5.opt.use_int8_inference=true;
            
            yolov5.load_param(param_path);//ncnn中load的参数只能是const char*,所以需要string.c_str()转换
            yolov5.load_model(bin_path);
        
        }//构造函数
        void detect(const cv::Mat& bgr,vector<Object>& objects);
    private:
        string model_path; //加载模型地址：xxx.param,xxx.bin
        float prob_threshold; //分类阈值参数
        ncnn::Net yolov5; //定义一个网络对象
};
#endif