#ifndef NET_Creator_H
#define NET_Creator_H
#include "layer.h"
#include "net.h"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <iostream>
#include <exception>
#include <time.h>
using namespace std;
struct Object
{
    cv::Rect_<float>rect;
    int label;
    float prob;
};
class Net_Creator
{
    public:
        ncnn::Net net;
        Net_Creator(string path,bool int8_inference):model_path_param(path) 
        {
            cout<<"--------Net_Creator初始化---------"<<endl;
            model_path_bin=path.replace(path.rfind("."),6,".bin");
            
        
            const char *param_path=model_path_param.c_str();
            const char *bin_path=model_path_bin.c_str();
            
   
            if(int8_inference)//是否使用int8推理
            {
//                 net.opt.use_vulkan_compute = true;//是否使用gpu加速
                cout<<"使用int8 inference"<<endl;
                net.opt.num_threads=8;
                net.opt.use_int8_inference=true; 
            }
            else
            {
                cout<<"不使用int8_inference"<<endl;
            }
            net.load_param(param_path);
            net.load_model(bin_path);
               
        }
        virtual void detect(const cv::Mat& bgr,string& rec){};
        virtual void detect(const cv::Mat& bgr,vector<Object>& objects){};
        
    private:
        string model_path_param,model_path_bin;
        bool int8_inference;
        
};



#endif