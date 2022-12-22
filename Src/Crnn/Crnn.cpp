#include"Crnn.h"
#include"Utils_crnn.h"
#include <string>
using namespace std;
string CRNN_Recognize::detect_crnn(cv::Mat& bgr)
{

//     const char *param_path=model_path_param.c_str();
//     const char *bin_path=model_path_bin.c_str();
    
    
// //     cout<<model_bin_path<<endl;
//     ncnn::Net crnn;
//     cout<<"加载crnn"<<endl;
    
//     //使用量化8推理模式
//     crnn.opt.num_threads=8;
//     crnn.opt.use_int8_inference=true;
    
//     crnn.load_param(param_path);
//     crnn.load_model(bin_path);
    
    const int dstHeight=32;
    const int dstWidth=280;
    
    int hi=bgr.rows;
    int wid=bgr.cols;
    
    float scale=32.0/(float)(hi);
    int wid_new=(int)(scale*wid);
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_RGB, bgr.cols, bgr.rows,wid_new, dstHeight);
    const float meanValues[3]={127.5,127.5,127.5};
    const float normValues[3]={1.0/127.5,1.0/127.5,1.0/127.5};
    
    in.substract_mean_normalize(meanValues,normValues);
    
    ncnn::Extractor ex=crnn.create_extractor();
    ex.input("input",in);
    
    ncnn::Mat out;
    ex.extract("output",out);
    
    //后处理
    return scoreToTextLine((float *) out.data, out.h, out.w);
    
}