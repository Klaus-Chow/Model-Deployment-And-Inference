#include "yolov5.h"
#include "layer.h"
#include "net.h"
#include "simpleocv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <map>
#include <list>
#include <numeric>
#include <fstream>
#include <cmath>
#include <exception>
#include <time.h>
#include <ctime>
#include <regex>
using namespace std;

// int detect_yolov5(const cv::Mat& bgr, std::string& model_param_path,std::vector<Object>& objects)
// {
//     ncnn::Net yolov5;

//     yolov5.opt.num_threads=8;
//     yolov5.opt.use_int8_inference=true;
    
// //     yolov5.opt.use_vulkan_compute = true;
//     // yolov5.opt.use_bf16_storage = true;

//     // original pretrained model from https://github.com/ultralytics/yolov5
//     // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models

//     //best_lite-sim_int8.param 身份证识别项目，身份证ROI的小模型
//     string model_param_path_ori=model_param_path;
//     string model_bin_path=model_param_path.replace(model_param_path.find("."),6,".bin");
//     const char *param_path=model_param_path_ori.c_str();
//     const char *bin_path=model_bin_path.c_str();
    
    
//     yolov5.load_param(param_path);//这里ncnn模型load的参数只能是const char *
//     yolov5.load_model(bin_path);

    
//     //image_process
//     const int img_max_size=1400;
//     int img_w = bgr.cols;
//     int img_h = bgr.rows;
//     int max_size=max(img_w,img_h);
//     float ratio_re=1.f;
    
//     //如果图像的尺寸大于1400，将其按照比例放缩
//     if(max_size>img_max_size)
//     {
//       ratio_re=(float)img_max_size/max_size;
//       img_w=(int)(ratio_re*img_w)/32*32;
//       img_h=(int)(ratio_re*img_h)/32*32;
    
//     }    
    
//     const int target_size = 640;
//     const float prob_threshold = 0.25f;
//     const float nms_threshold = 0.45f;

    

//     // letterbox pad to multiple of MAX_STRIDE
//     int w = img_w;
//     int h = img_h;
//     float scale = 1.f;
//     if (w > h)
//     {
//         scale = (float)target_size / w;
//         w = target_size;
//         h = h * scale;
//     }
//     else
//     {
//         scale = (float)target_size / h;
//         h = target_size;
//         w = w * scale;
//     }

//     ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

//     // pad to target_size rectangle
//     // yolov5/utils/datasets.py letterbox
//     int wpad = (w + 32 - 1) / 32 * 32 - w;
//     int hpad = (h + 32 - 1) / 32 * 32 - h;
//     ncnn::Mat in_pad;
//     ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

//     const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
//     in_pad.substract_mean_normalize(0, norm_vals);

//     ncnn::Extractor ex = yolov5.create_extractor();

//     ex.input("images", in_pad);

//     std::vector<Object> proposals;

//     // anchor setting from yolov5/models/yolov5s.yaml

//     // stride 8
//     {
//         ncnn::Mat out;
//         ex.extract("output", out);

//         ncnn::Mat anchors(6);
//         anchors[0] = 10.f;
//         anchors[1] = 13.f;
//         anchors[2] = 16.f;
//         anchors[3] = 30.f;
//         anchors[4] = 33.f;
//         anchors[5] = 23.f;

//         std::vector<Object> objects8;
//         generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

//         proposals.insert(proposals.end(), objects8.begin(), objects8.end());
//     }

//     // stride 16
//     {
//         ncnn::Mat out;

//         ex.extract("471", out);

//         ncnn::Mat anchors(6);
//         anchors[0] = 30.f;
//         anchors[1] = 61.f;
//         anchors[2] = 62.f;
//         anchors[3] = 45.f;
//         anchors[4] = 59.f;
//         anchors[5] = 119.f;

//         std::vector<Object> objects16;
//         generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

//         proposals.insert(proposals.end(), objects16.begin(), objects16.end());
//     }

//     // stride 32
//     {
//         ncnn::Mat out;

//         ex.extract("491", out);
        
//         ncnn::Mat anchors(6);
//         anchors[0] = 116.f;
//         anchors[1] = 90.f;
//         anchors[2] = 156.f;
//         anchors[3] = 198.f;
//         anchors[4] = 373.f;
//         anchors[5] = 326.f;

//         std::vector<Object> objects32;
//         generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

//         proposals.insert(proposals.end(), objects32.begin(), objects32.end());
//     }

//     // sort all proposals by score from highest to lowest
//     qsort_descent_inplace(proposals);

//     // apply nms with nms_threshold
//     std::vector<int> picked;
//     nms_sorted_bboxes(proposals, picked, nms_threshold);

//     int count = picked.size();

//     objects.resize(count);
//     for (int i = 0; i < count; i++)
//     {
//         objects[i] = proposals[picked[i]];

//         // adjust offset to original unpadded
//         float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
//         float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
//         float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
//         float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

//         // clip
//         x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
//         y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
//         x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
//         y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

//         objects[i].rect.x = x0;
//         objects[i].rect.y = y0;
//         objects[i].rect.width = x1 - x0;
//         objects[i].rect.height = y1 - y0;
//     }

//     return 0;
// }

