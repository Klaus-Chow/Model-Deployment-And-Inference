// #ifndef DBNET_H
// #define DBNET_H
// #include "layer.h"
// #include "net.h"
// #include <algorithm>
// #if defined(USE_NCNN_SIMPLEOCV)
// #include "simpleocv.h"
// #else
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #endif
// #include <stdio.h>
// #include <vector>
// #include <numeric>
// #include <fstream>
// #include <string>
// #include <iostream>
// #include <exception>
// #include <time.h>
// using namespace std;
// struct TextBox
// {
//     vector<cv::Point>boxPoint;
//     float score;
// };
// class DBNet_textdetect
// {
//     public:
//         DBNet_textdetect(string path,float bt,float t):model_path(path),boxThresh(bt),thresh(t)
//         {
//             cout<<"dbnet初始化"<<endl;
//             string model_param_path=model_path;
//             string model_bin_path=model_path.replace(model_param_path.find("."),6,".bin");
//             const char* param_path=model_param_path.c_str();
//             const char* bin_path=model_bin_path.c_str();
            
//             cout<<" "<<bin_path<<endl;
            
//             dbnet.load_param(param_path);
//             dbnet.load_model(bin_path);
//         }
//         void detect();//const cv::Mat& bgr,vector<TextBox>& rsbox
//     private:
//         string model_path;//加载模型地址
//         float boxThresh; // 识别阈值
//         float thresh; //nms
//         ncnn::Net dbnet;


// };
// #endif