// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//目前基于opencv 4.5.5

#include "layer.h"
#include "net.h"
#include "simpleocv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp" 
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
#include <fstream>
#include <sys/timeb.h>
// #include "hello.h"
// #include "operation.h"

using namespace std;
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);
                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}

static int detect_yolov5(const cv::Mat& bgr, std::string& model_param_path,std::vector<Object>& objects,float pro_threshold)
{
    ncnn::Net yolov5;

    yolov5.opt.num_threads=8;
    yolov5.opt.use_int8_inference=true;
    
//     yolov5.opt.use_vulkan_compute = true;
    // yolov5.opt.use_bf16_storage = true;

    // original pretrained model from https://github.com/ultralytics/yolov5
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models

    //best_lite-sim_int8.param 身份证识别项目，身份证ROI的小模型
    string model_param_path_ori=model_param_path;
    string model_bin_path=model_param_path.replace(model_param_path.find("."),6,".bin");
    const char *param_path=model_param_path_ori.c_str();
    const char *bin_path=model_bin_path.c_str();
    
    
    yolov5.load_param(param_path);//这里ncnn模型load的参数只能是const char *
    yolov5.load_model(bin_path);

    
    //image_process
//     const int img_max_size=1400;
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    int max_size=max(img_w,img_h);
    float ratio_re=1.f;
    
//     //如果图像的尺寸大于1400，将其按照比例放缩
//     cout<<"原图像大小："<<"w: "<<img_w<<" h: "<<img_h<<endl;
//     if(max_size>img_max_size)
//     {
//       ratio_re=(float)img_max_size/max_size;
// //       cout<<(int)(ratio_re*img_w)<<endl;
// //       cout<<(int)(ratio_re*img_h)<<endl;
//       img_w=(int)((ratio_re*img_w)/32*32);
//       img_h=(int)((ratio_re*img_h)/32*32);
    
//     }    
//     cout<<"大于1400大小修复："<<"w: "<<img_w<<" h: "<<img_h<<endl;
    const int target_size = 640;
    const float prob_threshold=pro_threshold;
    const float nms_threshold = 0.45f;

    

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
//     cout<<"按照640变换"<<h<<" "<<w<<endl;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 32 - 1) / 32 * 32 - w;
    int hpad = (h + 32 - 1) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output", out);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;

        ex.extract("471", out);

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;

        ex.extract("491", out);
        
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        
        "name","born_y","born_m","born_d","national","gender","location","ID","local","time","face"
//         "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//         "hair drier", "toothbrush"
    };
    cv::Mat image = bgr.clone();
//     cout<<objects.size()<<endl;
    
    for (int i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cout<<"save image"<<endl;
    cv::imwrite("./imgID_de.jpg",image);
   
}

//根据检测结果扣除图像化并且做了扩充，而且根据lable进行角度校正
static int GetCropImageAndIDType(cv::Mat& im_crop,std::string& ID_type,const cv::Mat im_ori,const Object& obj,const bool padding=true)
{
    
    int x_1,y_1,x_2,y_2,w_r,h_r,w_ori,h_ori,pad; 
    x_1=(int)obj.rect.x;
    y_1=(int)obj.rect.y;
    w_r=(int)obj.rect.width;
    h_r=(int)obj.rect.height;
    x_2=x_1+w_r;
    y_2=y_1+h_r;
    w_ori=im_ori.cols;
    h_ori=im_ori.rows;
    if(padding) //使得检测出来的ROI扩展一下
    {
        pad=min(w_r,h_r);
        x_1=max(0,x_1-pad/3);
        y_1=max(0,y_1-pad/3);
        x_2=min(w_ori-1,x_2+pad/3);
        y_2=min(h_ori-1,y_2+pad/3);
    
    }
    im_ori(cv::Rect(x_1, y_1,x_2-x_1, y_2-y_1)).copyTo(im_crop);
    
    int n_class=obj.label;
    if(n_class<4)
    {
    ID_type="obverse";
    }
    else
    {
    ID_type="reverse";
    }
    if(n_class==3 || n_class==7)
    {
        cv::transpose(im_crop,im_crop); //逆时针旋转90
        cv::flip(im_crop,im_crop,1);  //水平翻转
    }
    else if(n_class == 2 || n_class ==6)
    {
        cv::flip(im_crop,im_crop,-1);  //水平垂直翻转
    
    }
    else if(n_class == 1 || n_class ==5)
    {
    
        cv::transpose(im_crop,im_crop); //逆时针旋转90
        cv::flip(im_crop,im_crop,0);  //垂直翻转
    }   
    return 0;
}

//
static cv::Mat get_perspective_image(cv::Mat im_c,cv::Mat& coord)
{
    cv::Mat im_p;
    float width,height;
    width=640.0;
    height=416.0;
    float a[4][2]={{0.0,0.0},{width,0.0},{width,height},{0.0,height}};
    cv::Mat pts1(4,2,CV_32FC1,a);
    cv::Mat pts2 = coord.reshape(2,4);
//     cout<<pts1<<endl;
//     cout<<pts2<<endl;
    cv::Mat H=cv::getPerspectiveTransform(pts2,pts1);
    cv::warpPerspective(im_c,im_p,H,cv::Size(640,416));
    return im_p;
}


template<typename T>
T SumVector(vector<T>& vec)
{
    T res = 0;
    for (size_t i=0; i<vec.size(); i++)
    {
        res += vec[i];
    }
    return res;
}

void linalgsolve(const float A[4],const float b[2],float result[2],bool &get_kb)
{
    try
    {
        result[1] = (A[0]*b[1] - A[2]*b[0])/((A[0]*A[3])-(A[2]*A[1]));
        result[0] = (b[0] - A[1]*result[1])/A[0];
        get_kb = true; 
        }
    catch(exception e){
        get_kb = false; 
    }  
}
float* linear_regression(vector<vector<float>>& line,bool &get_kb)
{
    float *result = new float[2];
    int N = line.size();
    vector<float>x(N);
    vector<float>x2(N);
    vector<float>y(N);
    vector<float>xy(N);
    float b[2];
    float A[4];
    for(int i=0;i<N;i++) 
    {
        x[i] = line[i][0];
        y[i] = line[i][1];
        x2[i] = x[i]*x[i];
        xy[i] = x[i]*y[i];
//         cout << "x: " << x2[i] << " ";
    }   
    b[0] = SumVector(y); //(float) accumulate(y.begin(), y.end(), 0);
    b[1] = SumVector(xy); //accumulate(xy.begin(), xy.end(), 0);
    A[0] = N;
    A[1] = SumVector(x); // accumulate(x.begin(), x.end(), 0);
    A[2] = SumVector(x); //accumulate(x.begin(), x.end(), 0);
    A[3] = SumVector(x2); //accumulate(x2.begin(), x2.end(), 0);

//     result[1] = (A[0]*b[1] - A[2]*b[0])/((A[0]*A[3])-(A[2]*A[1]));
//     result[0] = (b[0] - A[1]*result[1])/A[0];
    linalgsolve(A,b,result,get_kb);    
    return result;
}

static std::vector<float> crosspoint(float *k1_b1,float *k2_b2,float x1,float x3,bool target_r1,bool target_r2,bool& point_is_exist)
{
    vector<float>point;
    point_is_exist=false;
    if(target_r1 == false)
    {
        if(target_r2)
        {
          float x=x1;
          float y=k2_b2[1]*x1+k2_b2[0];
          point.push_back(x);
          point.push_back(y);
          point_is_exist=true;
            
        }
    }
    else if(target_r2==false)
    {
        float x=x3;
        float y=k1_b1[1]*x3+k1_b1[0];
        point.push_back(x);
        point.push_back(y);
    }
    else if(k1_b1[1]!=k2_b2[1])
    {
        float x=(k2_b2[0]-k1_b1[0])/(k1_b1[1]-k2_b2[1]);
        float y=k1_b1[1]*x+k1_b1[0];
        point.push_back(x);
        point.push_back(y);
        point_is_exist=true;
    }
    return point;
}

// 有效期限 template_arr[0];签发机关[1];居民身份证[2];中华人民共和国[3]
bool JudgeImgPSuccessF(const cv::Mat &template_img, const cv::Mat &result){
    
    int template_arr[4][4] = {{197, 456, 313, 486},{197, 389, 313, 419},
                                     {255, 149, 779, 224},{283, 49, 745, 112}
                                    };
    for (int i = 0; i < 4; i++){
        template_arr[i][0] = (int) template_arr[i][0]/1.3375;
        template_arr[i][1] = (int) template_arr[i][1]/1.2981;
        template_arr[i][2] = (int) template_arr[i][2]/1.3375;
        template_arr[i][3] = (int) template_arr[i][3]/1.2981;      
    }

    int flag = 0;
    for (int j = 0; j < 4; j++){ 
        cv::Mat cut_template,cut_result;
        cut_template = template_img(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
        cut_result = result(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
        cv::cvtColor(cut_template, cut_template,cv::COLOR_RGB2GRAY);
        cv::cvtColor(cut_result, cut_result,cv::COLOR_RGB2GRAY);
        cv::threshold(cut_template,cut_template, 0, 255, cv::THRESH_OTSU);
        cv::threshold(cut_result,cut_result, 0, 255, cv::THRESH_OTSU);
        cut_template.convertTo(cut_template, CV_32F);
        cut_result.convertTo(cut_result, CV_32F);
        cut_template = cut_template - cut_result;
        cut_template.setTo(1,cut_template < 0);
        cut_template.convertTo(cut_template, CV_32F);
        cv::Scalar get_mean;
        get_mean = cv::mean(cut_template);
//         std::cout << "cut_template sum： " << cv::sum(cut_template) << std::endl;
//         std::cout << "cut_template mean： " << cv::mean(cut_template) << std::endl; 

        if (get_mean[0] < 15){
            flag += 1;
            } 
    }
    if (flag >= 2){
        return true;
        }
    return false;      
}
    
// 年 template_arr[0];月[1];日[2];姓名[3];性别[3];民族[3];出生[3];住址[3];身份证号码[3]
bool JudgeImgPSuccessZ(const cv::Mat &template_img, const cv::Mat &result, int numthresh=5, int meanthresh=15){

    
    int template_arr[9][4] = {{260,208,291,239},{337,208,368,239},
                              {416,208,447,239},{70,81,143,112},
                              {70,142,143,173},{259,144,332,175},
                              {68,208,141,239},{71,278,142,309},
                              {70,449,261,481}
                                    };
    for (int i = 0; i < 9; i++){
        template_arr[i][0] = (int) template_arr[i][0]/1.3375;
        template_arr[i][1] = (int) template_arr[i][1]/1.2981;
        template_arr[i][2] = (int) template_arr[i][2]/1.3375;
        template_arr[i][3] = (int) template_arr[i][3]/1.2981;      
    }

    int flag = 0;
    for (int j = 0; j < 9; j++){ 
        cv::Mat cut_template,cut_result;
        cut_template = template_img(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
        cut_result = result(cv::Range(template_arr[j][1],template_arr[j][3]),cv::Range(template_arr[j][0],template_arr[j][2])).clone(); 
        cv::cvtColor(cut_template, cut_template,cv::COLOR_RGB2GRAY);
        cv::cvtColor(cut_result, cut_result,cv::COLOR_RGB2GRAY);
        cv::threshold(cut_template,cut_template, 0, 255, cv::THRESH_OTSU);
        cv::threshold(cut_result,cut_result, 0, 255, cv::THRESH_OTSU);
        cut_template.convertTo(cut_template, CV_32F);
        cut_result.convertTo(cut_result, CV_32F);
        cut_template = cut_template - cut_result;
        cut_template.setTo(1,cut_template < 0);
        cv::Scalar get_mean;
        get_mean = cv::mean(cut_template);
//         std::cout << "cut_template sum： " << cv::sum(cut_template) << std::endl;
//         std::cout << "cut_template mean： " << cv::mean(cut_template) << std::endl; 
        if (get_mean[0] < meanthresh){
            flag += 1;
            } 
    }
    if (flag >= numthresh){
        return true;
        }
    return false;      
}

//边缘检测模型，根据得到的点的类型对直线拟合校正目标
static int GetImgByEdge(cv::Mat& im_p,std::string& ID_type,const cv::Mat im_c,std::vector<Object>& edgeCoords,bool& perspective)
{

    perspective=false;
    typedef map<char, vector<vector<float>>> Eclass;
    Eclass eclass; 
    for(int i=0;i<edgeCoords.size();i++)
    {
        vector<float> xy_c;
        const Object& obj = edgeCoords[i];
        char key=obj.label+'0'; //int转换成char类型
        int x_1=(int)obj.rect.x;
        int y_1=(int)obj.rect.y;
        int w_r=(int)obj.rect.width;
        int h_r=(int)obj.rect.height;
        int x_2=x_1+w_r;
        int y_2=y_1+h_r;
        float x_c=(x_1+x_2)/2.0;
        float y_c=(y_1+y_2)/2.0;
        xy_c.push_back(x_c);
        xy_c.push_back(y_c);
        eclass[key].push_back(xy_c);  
    }

    
//     //打印出来看看
//     cout<<"label is 0:"<<eclass['0'].size()<<endl;
//     cout<<"label is 1:"<<eclass['1'].size()<<endl;
//     cout<<"label is 2:"<<eclass['2'].size()<<endl;
//     cout<<"label is 3:"<<eclass['3'].size()<<endl;
    
//     for(Eclass::iterator it=eclass.begin();it!=eclass.end();it++)
//     {
//        cout << "key:"<<it->first;
//        for (vector<vector<float>>::iterator itv = (it->second).begin();itv != (it->second).end();itv++)//通过迭代器访问数据
//         {
// // 		//*it 代表的是vector<int>，又是一个容器，所以需要把小容器里边的数据遍历出来
// // 		//小技巧：*it 就是vector 尖括号 里边的数据类型
// //             cout<<"num of vector"<<(it->second).size()<<endl;;
//             for (vector<float>::iterator vit = (*itv).begin();vit != (*itv).end();vit++)
//             {
//                 cout<<" "<<*vit <<" ";
//             }
//         }
//        cout<<"----------------------------------"<<endl;

//     }    
    
    if((int)(eclass['0'].size())==1 && (int)(eclass['1'].size())==1 && (int)(eclass['2'].size())==1 && (int)(eclass['3'].size())==1)
    {
        vector<float>coord;
        for(int i=0;i<4;i++)
        {
            char c=i+'0';
            for (vector<vector<float>>::iterator itv =eclass[c].begin();itv != eclass[c].end();itv++)//通过迭代器访问数据
            {
                for (vector<float>::iterator vit = (*itv).begin();vit != (*itv).end();vit++)
                {
                    coord.push_back(*vit);
                }
            }
        
        } 
        cv::Mat coord_point=cv::Mat(coord,true); //8*1
        im_p=get_perspective_image(im_c,coord_point);
        perspective=true;
//         cv::imwrite("./im_p.jpg",im_p);
        return 0;
    }
    if(((int)(eclass['0'].size())+(int)(eclass['1'].size())+(int)(eclass['4'].size()))>=2 && ((int)(eclass['1'].size())+(int)(eclass['2'].size())+
      (int)(eclass['6'].size()))>=2 && ((int)(eclass['2'].size())+(int)(eclass['3'].size())+(int)(eclass['5'].size()))>=2 && ((int)(eclass['3'].size())+
      (int)(eclass['0'].size())+(int)(eclass['7'].size()))>=2)
    {
        
    
        vector<vector<float>>line1,line2,line3,line4;
        
        if((int)(eclass['0'].size())==1)
        {
            line1.push_back(eclass['0'][0]);
        }
        if((int)(eclass['1'].size())==1)
        {
            line1.push_back(eclass['1'][0]);
        }
        
        for(int i=0;i<(int)(eclass['4'].size());i++)
        {
            line1.push_back(eclass['4'][i]);
        
        }
        
         if((int)(eclass['1'].size())==1)
        {
            line2.push_back(eclass['1'][0]);
        }
        if((int)(eclass['2'].size())==1)
        {
            line2.push_back(eclass['2'][0]);
        }
        
        for(int i=0;i<(int)(eclass['6'].size());i++)
        {
            line2.push_back(eclass['6'][i]);
        
        }
        
          if((int)(eclass['2'].size())==1)
        {
            line3.push_back(eclass['2'][0]);
        }
        if((int)(eclass['3'].size())==1)
        {
            line3.push_back(eclass['3'][0]);
        }
        
        for(int i=0;i<(int)(eclass['5'].size());i++)
        {
            line3.push_back(eclass['5'][i]);
        
        }
        
          if((int)(eclass['3'].size())==1)
        {
            line4.push_back(eclass['3'][0]);
        }
        if((int)(eclass['0'].size())==1)
        {
            line4.push_back(eclass['0'][0]);
        }
        
        for(int i=0;i<(int)(eclass['7'].size());i++)
        {
            line4.push_back(eclass['7'][i]);
        
        }
//         cout<<line1.size()<<" "<<line2.size()<<" "<<line3.size()<<" "<<line4.size()<<endl;
        
//         cout<<line1[0][0]<<endl;
//         cout<<line1[1][0]<<endl;
        if(int(line1.size())>1 && int(line2.size())>1 && int(line3.size())>1 &&int(line4.size())>1)
        {
            vector<float>angleC_4;
            vector<float>point;
            vector<vector<float>>angleCoords;
            bool target_r1=false,target_r2=false,target_r3=false,target_r4=false;   
            float *k1_b1=linear_regression(line1,target_r1); //线性会归根据坐标点拟合出直线
            float *k2_b2=linear_regression(line2,target_r2);
            float *k3_b3=linear_regression(line3,target_r3);
            float *k4_b4=linear_regression(line4,target_r4);

//             cout<<"k1:"<<k1_b1[1]<<"b1:"<<k1_b1[0]<<endl;
//             cout<<"k2:"<<k2_b2[1]<<"b2:"<<k2_b2[0]<<endl;
//             cout<<"k3:"<<k3_b3[1]<<"b3:"<<k3_b3[0]<<endl;
//             cout<<"k4:"<<k4_b4[1]<<"b4:"<<k4_b4[0]<<endl;
            bool point_is_exist=false;
            point= crosspoint(k1_b1,k4_b4,line1[0][0],line1[1][0],target_r1,target_r4,point_is_exist);
            if(point_is_exist)
            {
                angleCoords.push_back(point);
            }
            point= crosspoint(k1_b1,k2_b2,line2[0][0],line2[1][0],target_r1,target_r2,point_is_exist);
            if(point_is_exist)
            {
                angleCoords.push_back(point);
            }
            point= crosspoint(k2_b2,k3_b3,line3[0][0],line3[1][0],target_r2,target_r3,point_is_exist);
            if(point_is_exist)
            {
                angleCoords.push_back(point);
            }
            point= crosspoint(k3_b3,k4_b4,line4[0][0],line4[1][0],target_r3,target_r4,point_is_exist);
            if(point_is_exist)
            {
                angleCoords.push_back(point);
            }
           for(int i=0;i<(int)(angleCoords.size());i++)
           {
               for(int j=0;j<int(angleCoords[i].size());j++)
               {
                    angleC_4.push_back(angleCoords[i][j]);
               }
           }
           if((int)(angleC_4.size())==8)
            {
                cv::Mat angleC_4_mat=cv::Mat(angleC_4,true); //8*1
                cv::Mat im_an=get_perspective_image(im_c,angleC_4_mat);
//                 cv::imwrite("./im_angle.jpg",im_an);
                cv::Mat template_img;
                if(ID_type=="obverse")
                {
                   template_img=cv::imread("/src/notebooks/ImgIDRecogProj/utils/templateZ.jpg",1);
                }
               else
               {
                   template_img=cv::imread("/src/notebooks/ImgIDRecogProj/utils/templateF.jpg",1);
               }
               if(ID_type=="obverse")
               {
                   if(JudgeImgPSuccessZ(im_an,template_img))
                   {
                       perspective=true;
                   }
               }
               else
               {
                   if(JudgeImgPSuccessF(im_an,template_img))
                   {
                       perspective=true;
                   }
                   
               }
               return 0;
               
            }

        }
 
    }
    return 0;
}

//将yolo预测的坐标x,y,w,h转换成文本识别的四个点（1*8），因为是yolo的检测所以输出需要改变
static int fixbox(std::vector<vector<float>>& textbboxes,std::vector<int>& nclasses,std::vector<Object>& textbbox)
{
    for(int i=0;i<(int)(textbbox.size());i++)
    {
        vector<float> point_x1y1_x2y2;
        const Object& textobj = textbbox[i];
        float x_1=textobj.rect.x;
        float y_1=textobj.rect.y;
        float w_r=textobj.rect.width;
        float h_r=textobj.rect.height;
        float x_2=x_1+w_r;
        float y_2=y_1+h_r;
        point_x1y1_x2y2.push_back(x_1);
        point_x1y1_x2y2.push_back(y_1);
        point_x1y1_x2y2.push_back(x_2);
        point_x1y1_x2y2.push_back(y_1);
        point_x1y1_x2y2.push_back(x_2);
        point_x1y1_x2y2.push_back(y_2);
        point_x1y1_x2y2.push_back(x_1);
        point_x1y1_x2y2.push_back(y_2);
        
        textbboxes.push_back(point_x1y1_x2y2);
        nclasses.push_back(textobj.label);
    }
    return 0;
}



//从txt文本读取中文汉字(用ifstream读取txt文件后得到string类型，再用该函数切出每个汉字)
std::vector<std::string> split_chinese(std::string s) {
    std::vector<std::string> t;
    for (size_t i = 0; i < s.length();)
    {
        int cplen = 1;
        if ((s[i] & 0xf8) == 0xf0)      // 11111000, 11110000
            cplen = 4;
        else if ((s[i] & 0xf0) == 0xe0) // 11100000
            cplen = 3;
        else if ((s[i] & 0xe0) == 0xc0) // 11000000
            cplen = 2;
        if ((i + cplen) > s.length())
            cplen = 1;
        t.push_back(s.substr(i, cplen));
        i += cplen;
    }
    return t;
}


std::string scoreToTextLine(const float *outputData, int h, int w)
{
    //加载文本txt
    string s;
    ifstream inf;
//     inf.open("/src/notebooks/ncnn-20220420/build/key_repvgg.txt");
    inf.open("/src/notebooks/crnnmobile/key.txt");
    getline(inf,s);
    std::vector<std::string>keys = split_chinese(s);
    keys.push_back("-");
//     cout<<(int)(keys.size())<<endl;
//     std::vector<std::string> keys={"0","1","2","3","4","5","6","7","8","9"};
    int keySize =(int)(keys.size());
//     cout<<keySize<<endl;
//     cout<<keys[0]<<endl;
    std::string strRes;
    std::vector<float> scores;
    int lastIndex = 0;
    int maxIndex;
    float maxValue;


    for (int i = 0; i < h; i++)
    {
        maxIndex = 0;
        maxValue = -1000.f;
        //do softmax
        std::vector<float> exps(w);
        for (int j = 0; j < w; j++) {
            float expSingle = exp(outputData[i * w + j]);
//             printf("%f\n",expSingle);
            exps.at(j) = expSingle;
        }
        float partition = accumulate(exps.begin(), exps.end(), 0.0);//row sum
        for (int j = 0; j < w; j++) {
            float softmax = exps[j] / partition;
            if (softmax > maxValue) {
                maxValue = softmax;
                maxIndex = j;
                
            }
        }
       if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(keys[maxIndex - 1]);
        }
        lastIndex = maxIndex;
    }
    
    return strRes;
}
static std::string detect_crnn(const cv::Mat& bgr, string model_param_path,std::vector<float>& cls_scores)
{
    std::string Res;
    ncnn::Net crnn;
    crnn.opt.num_threads=8;
    crnn.opt.use_int8_inference=true;
    string model_param_path_ori=model_param_path;
    string model_bin_path=model_param_path.replace(model_param_path.find("."),6,".bin"); //从.开始长度为6即.param替换成.bin
    const char *param_path=model_param_path_ori.c_str();
    const char *bin_path=model_bin_path.c_str();
    
    
//     crnn.opt.use_vulkan_compute = true;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    
    //模型1：crnn_moblie_op11_sim_fp16.param，crnn_moblie_op11_sim_fp16.bin
    //模型2：crnn_moblie_mobilev3-augTrue_sim_fp16
    //模型3：BaiDu_netCRNN_fp16
    
    crnn.load_param(param_path);
    crnn.load_model(bin_path);

    const int dstHeight = 32;
    const int dstWidth = 280;
    
    
    int hi=bgr.rows; //h
    int wid=bgr.cols; //w
    
//     cout<<hi<<" "<<wid<<endl;
    
    
    float scale=32.0/(float)(hi);
//     cout<<scale<<endl;
    int wid_new=(int)(scale*wid);
//     cout<<wid_new<<endl;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_RGB, bgr.cols, bgr.rows,wid_new, dstHeight);

    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    
    in.substract_mean_normalize(meanValues, normValues);
//     fprintf(stderr, "h=%d,w=%d,c=%d\n", in.h, in.w,in.c);
    
    ncnn::Extractor ex = crnn.create_extractor();

    ex.input("input", in);
//     cout<<in.h<<" "<<in.w<<" "<<in.c<<endl;
//      for (int q=0;q<1;q++)
//     {
//        const float* ptr = in.channel(q);
//        for (int y=0; y<in.h; y++)
//         {
//             for (int x=0; x<in.w; x++)
//             {
//                 printf("%f ", ptr[x]);
//             }
//             ptr += in.w;
//             printf("\n");
//         }
//         printf("------------------------\n");  
//         break;
//     }
    
    
    ncnn::Mat out;
    ex.extract("output", out);
//     for (int q=0;q<1;q++)
//     {
//        const float* ptr = out.channel(q);
//        for (int y=0; y<out.h; y++)
//         {
//             for (int x=0; x<out.w; x++)
//             {
//                 printf("%f ", ptr[x]);
//             }
//             ptr += out.w;
//             printf("\n");
//         }
//         printf("------------------------\n");  
//         break;
//     }
//     cout<<out.h<<" "<<out.w<<endl;
    Res=scoreToTextLine((float *) out.data, out.h, out.w);
    
    
    return Res;
}





//text_detect后扣取图像部分三个相关函数
static float cal_distance(float x1,float y1,float x2,float y2)
{
    return sqrt(pow((x2-x1),2)+pow((y2-y1),2));

}
static std::pair<float,float> cal_width_height(cv::Mat coord)
{
    
    std::pair<float,float>wh;
    float width,height;
    float x1,y1,x2,y2,x3,y3;
    x1=coord.at<float>(0,0);
    y1=coord.at<float>(1,0);
    x2=coord.at<float>(2,0);
    y2=coord.at<float>(3,0);
    x3=coord.at<float>(4,0);
    y3=coord.at<float>(5,0);
    
    width=cal_distance(x1,y1,x2,y2);
    height=cal_distance(x2,y2,x3,y3);
//     cout<<"cal_width_height"<<endl;
//     cout<<width<<" "<<height<<endl;
    wh.first=width;
    wh.second=height;
    
    return wh;
}
static cv::Mat get_perspective_image_rec_result(cv::Mat im_c,cv::Mat& coord)
{
    
    cv::Mat im_p;
    float width,height;
    std::pair<float,float>wh;
    wh=cal_width_height(coord);
    width=wh.first;
    height=wh.second;
//     cout<<width<<" "<<height<<endl;
    float a[4][2]={{0.0,0.0},{width,0.0},{width,height},{0.0,height}};
    cv::Mat pts1(4,2,CV_32FC1,a);
    cv::Mat pts2 = coord.reshape(2,4);
//     cout<<pts1<<endl;
//     cout<<pts2<<endl;
    cv::Mat H=cv::getPerspectiveTransform(pts2,pts1);
    cv::warpPerspective(im_c,im_p,H,cv::Size((int)(width),(int)(height)));
//     cv::imwrite("im_ppppppppp.jpg",im_p);
    return im_p;

}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
  // 初始化索引向量
  std::vector<size_t> idx(v.size());
  //使用iota对向量赋0~？的连续值
  std::iota(idx.begin(), idx.end(), 0);
  // 通过比较v的值对索引idx进行排序
  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

vector<string> split_utf8(string str){
    string strChar;
    vector<string> str_s;
    for(int i = 0; str[i] != '\0'; )
    {
        char chr = str[i];
//         cout << chr <<endl;
        if((chr & 0x80) == 0)
        {
            strChar = str.substr(i,1);
            ++i;
        }//chr是1111 1xxx
        else if((chr & 0xF8) == 0xF8)
        {
            strChar = str.substr(i, 5);
            i+=5;
        }//chr是1111 xxxx
        else if((chr & 0xF0) == 0xF0)
        {
            strChar = str.substr(i, 4);
            i+=4;
        }//chr是111x xxxx
        else if((chr & 0xE0) == 0xE0)
        {
            strChar = str.substr(i, 3);
            i+=3;
        }//chr是11xx xxxx
        else if((chr & 0xC0) == 0xC0)
        {
            strChar = str.substr(i, 2);
            i+=2;
        }
        str_s.push_back(strChar);
    }
    return str_s;
}

struct ID_Result_Obverse
{
     //成员列表
    string name="";
    vector<float>name_coord={0,0,0,0,0,0,0,0};
    
    string national="";
    vector<float>national_coord={0,0,0,0,0,0,0,0};
    
    string sex="";
    vector<float>sex_coord={0,0,0,0,0,0,0,0};
    
    string year="";
    vector<float>year_coord={0,0,0,0,0,0,0,0};
    
    string month="";
    vector<float>month_coord={0,0,0,0,0,0,0,0};

    string day="";
    vector<float>day_coord={0,0,0,0,0,0,0,0};
    
    string birth="";
    vector<float>birth_coord={0,0,0,0,0,0,0,0};
    
    string address="";
    vector<float>address_coord={0,0,0,0,0,0,0,0};
    
    string id_number="";
    vector<float>id_number_coord={0,0,0,0,0,0,0,0}; 
    
    string id_type="obverse";
};

struct ID_Result_Reverse
{
     //成员列表
    string issued="";
    vector<float>issued_coord={0,0,0,0,0,0,0,0};
    
    string effective_date="";
    vector<float>effective_date_coord={0,0,0,0,0,0,0,0};
    
    string id_type="reverse";
};

void nclassindex(vector<int>& vec,vector<int>& res,int nclass)
{
    for(int i=0;i<vec.size();i++)
    {
        if(vec[i]==nclass)
        {
         res.push_back(i);
        }
    
    }
}


//正则化匹配
string matchString(string pattern_get,string content)
{
    
    regex pattern(pattern_get);
    smatch matchResult;  //正则化匹配string类型
    string matchString="";
    //迭代器声明
    string::const_iterator iterStart = content.begin();
    string::const_iterator iterEnd = content.end();

    while (regex_search(iterStart, iterEnd, matchResult, pattern))
    {
        matchString += matchResult[0];
        iterStart = matchResult[0].second;	//更新搜索起始位置,搜索剩下的字符串
    } 
    return matchString;

}

// # 单字民族对应的相似字符, 用于辅助纠正
void get_similar_sex(map<string,vector<string>>& similar_sex){
    vector<string>similar_sex1={"墨","奥","速","隽","贵","美","期","勇"};
    similar_sex.insert(pair<string,vector<string>>("男",similar_sex1));//["男"].push_back(similar_sex1);
    vector<string>similar_sex2={"因","安","汝","文","委"};
    similar_sex.insert(pair<string,vector<string>>("女",similar_sex2));//["女"].push_back(similar_sex2);
//     cout<<"sex: "<<similar_sex["男"].size()<<" ";    
}

void get_similar(map<string,vector<string>>& similar_data_nation,vector<string>&nationality){
    
    nationality={"汉","彝","侗","蒙古","回","藏","维吾尔",
                  "苗","壮","朝鲜","满","瑶","白","土家",
                  "哈尼","哈萨克","黎","傈僳","佤","畲","高山",
                  "拉祜","水","东乡","纳西","景颇","柯尔克孜",
                  "土","达斡尔","羌","撒拉","毛难","仫佬",
                  "仡佬","锡伯","阿昌","普米","塔吉克","怒","乌孜别克",
                  "俄罗斯","德昂","保安","裕固","崩龙","独龙","鄂伦春","赫哲",
                  "门巴","珞巴","基诺","鄂温克","傣","京","塔塔尔","布朗","布依"
                  };   
    
    vector<string>similar1={"固", "同", "倜", "调", "垌", "桐", "恫", "洞", "峒", "硐", "胴"};
    similar_data_nation.insert(pair<string,vector<string>>("侗",similar1));//["侗"].push_back(similar1);
    vector<string>similar2={"瞒", "蹒", "螨", "潢", "滿"};
    similar_data_nation.insert(pair<string,vector<string>>("满",similar2));//["满"].push_back(similar2);
    vector<string>similar3={"汶", "汊", "汝", "汐", "汲", "汀", "波", "叹", "仅", "汊"};
    similar_data_nation.insert(pair<string,vector<string>>("汉",similar3));//["汉"].push_back(similar3);
    vector<string>similar4={"惊", "凉", "谅", "掠", "谅", "晾", "掠", "景", "亰"};
    similar_data_nation.insert(pair<string,vector<string>>("京",similar4));//["京"].push_back(similar4);
    vector<string>similar5={"泰", "秦", "倴", "僚", "溙", "奉"};
    similar_data_nation.insert(pair<string,vector<string>>("傣",similar5));//["傣"].push_back(similar5);

    vector<string>similar6={"亩", "启", "茔", "芸", "盅", "电", "宙", "田", "喵", "描", "猫"};
    similar_data_nation.insert(pair<string,vector<string>>("苗",similar6));//["苗"].push_back(similar6);
    vector<string>similar7={"臧", "臧", "葬"};
    similar_data_nation.insert(pair<string,vector<string>>("藏",similar7));//["壮"].push_back(similar7);
    vector<string>similar8={"壯", "莊", "荘", "妆"};
    similar_data_nation.insert(pair<string,vector<string>>("壮",similar8));//["壮"].push_back(similar7);


    vector<string>similar9={"摇", "遥", "谣"};
    similar_data_nation.insert(pair<string,vector<string>>("瑶",similar9));//["瑶"].push_back(similar8);
    vector<string>similar10={"日", "自", "百", "囱", "曰", "囪", "甶", "凹", "汩", "彐",
              "旧", "囗", "田", "帕", "伯", "拍", "泊", "柏", "陌"};
    similar_data_nation.insert(pair<string,vector<string>>("白",similar10));//["白"].push_back(similar9);

    vector<string>similar11={"藜", "棃", "黧", "梨", "犁", "藜"};
    similar_data_nation.insert(pair<string,vector<string>>("黎",similar11));//["黎"].push_back(similar10);
    vector<string>similar12={"仉", "巩", "讥", "伉", "瓦", "咓", "砙"};
    similar_data_nation.insert(pair<string,vector<string>>("佤",similar12));//["佤"].push_back(similar11);

    vector<string>similar13={"番", "禽", "肏"};
    similar_data_nation.insert(pair<string,vector<string>>("畲",similar13));//["畲"].push_back(similar12);
    vector<string>similar14={"氷", "囦", "永", "冰", "木", "未"};
    similar_data_nation.insert(pair<string,vector<string>>("水",similar14));//["水"].push_back(similar13);

    vector<string>similar15={"工", "二", "三", "王", "亍", "士", "七"};
    similar_data_nation.insert(pair<string,vector<string>>("土",similar15));//["土"].push_back(similar14);
    vector<string>similar16={"恙", "羊", "羔", "羔"};
    similar_data_nation.insert(pair<string,vector<string>>("羌",similar16));//["羌"].push_back(similar15);

    vector<string>similar17={"囚", "四", "迴", "佪", "廻", "洄", "叵", "固", "国","间", "囧", "区","而"};
    similar_data_nation.insert(pair<string,vector<string>>("回",similar17));//["回"].push_back(similar17);
    vector<string>similar18={"努", "恕", "奴", "弩", "驽", "孥", "㐐"};
    similar_data_nation.insert(pair<string,vector<string>>("怒",similar18));//["怒"].push_back(similar18);   
}



string judge_nationality(string data_n){
    string result = "";
    map<string,vector<string>> similar_data_nation;
    vector<string>nationality;
    get_similar(similar_data_nation,nationality);  
//     cout<<"nationality: "<<nationality[0]<<" ";
    vector<float> scores;
    
    if (count(nationality.begin(),nationality.end(),data_n)){
        return data_n;
    }
    else{
        vector<string>data_n_s = split_utf8(data_n);
        if (data_n_s.size() == 1)
        {
            for(map<string,vector<string>>::iterator iter = similar_data_nation.begin(); iter != similar_data_nation.end(); ++iter)
            {
                string nation_key = iter->first;
//                 cout<<"key:"<<nation_key;
                vector<string>similar_nation = iter->second;
                for (int i=0;i<similar_nation.size();i++)
                {
//                     cout<<similar_nation[i]<<" "; 
                    if (similar_nation[i]==data_n)
                    {
//                         cout<<"modify:"<<nation_key<<" ";
                        return nation_key;
                    }
                }
//                 cout<<endl;
            }
        }
        for (int num=0;num<nationality.size();num++)
        {
            vector<string>nation = split_utf8(nationality[num]);
            float score = 0;
            for (int idx_d=0;idx_d<data_n_s.size();idx_d++)
            {
                for(int idx_n=0;idx_n<nation.size();idx_n++)
                {
                    if (nation[idx_n] == data_n_s[idx_d])
                    {
                        if (idx_d == idx_n)
                        {
                            score += 100;
                        }
                        else
                        {
                            score += 50;
                        }
                        break;
                    }
                }  
            }
            if (data_n_s.size() == nation.size())
            {
                score += 70;
            }
            scores.push_back(score);
        }
    }
    auto index_ = sort_indexes(scores);
    return nationality[index_[index_.size()-1]];
}


string fix_sex(string sex_str)
{
    if (sex_str=="男" || sex_str=="女")
    {
        return sex_str;
    }
    else{
        map<string,vector<string>> similar_sex;
        get_similar_sex(similar_sex);
        for(map<string,vector<string>>::iterator iter = similar_sex.begin(); iter != similar_sex.end(); ++iter)
        {
            string sex_key = iter->first;
//             cout<<"key:"<<sex_key;
            vector<string>similar_sex = iter->second;
            for (int i=0;i<similar_sex.size();i++)
            {
//                 cout<<similar_sex[i]<<" "; 
                if (similar_sex[i]==sex_str)
                {
//                     cout<<"modify sex:"<<sex_key<<" ";
                    return sex_key;
                }
            }
        }        
      }
    
}
void get_result_observe(ID_Result_Obverse *Obverse_result,vector<int>&nclasses,vector<vector<float> >& bbox,vector<string>&reg_result)
{
    if (reg_result.size()>0){
        //name
        vector<int>index_class0;
        nclassindex(nclasses,index_class0,0);
        if (index_class0.size()==1){
            string class0_pattern = "[a-zA-Z0-9\u4e00-\u9fa5·]+";
            Obverse_result->name = matchString(class0_pattern,reg_result[index_class0[0]]);
            Obverse_result->name_coord = bbox[index_class0[0]];
        }
        
        //year
        vector<int>index_class1;
        nclassindex(nclasses,index_class1,1);     
        if (index_class1.size()==1){
            string class1_pattern = "[0-9]+";
            Obverse_result->year = matchString(class1_pattern,reg_result[index_class1[0]]); 
            Obverse_result->year_coord = bbox[index_class1[0]];
        }            

        //month
        vector<int>index_class2;
        nclassindex(nclasses,index_class2,2);     
        if (index_class2.size()==1){
            string class2_pattern = "[0-9]+";
            Obverse_result->month = matchString(class2_pattern,reg_result[index_class2[0]]); 
            Obverse_result->month_coord = bbox[index_class2[0]];
        } 
        
        //day
        vector<int>index_class3;
        nclassindex(nclasses,index_class3,3);     
        if (index_class3.size()==1){
            string class3_pattern = "[0-9]+";
            Obverse_result->day = matchString(class3_pattern,reg_result[index_class3[0]]); 
            Obverse_result->day_coord = bbox[index_class3[0]];
        }
        
        //birth
        Obverse_result->birth = Obverse_result->year+"."+Obverse_result->month+"."+Obverse_result->day;
        vector<vector<float> > birth_all(8,vector<float>(3,0));
        
        for(int i=0;i<birth_all.size();i++){
           birth_all[i][0] = Obverse_result->year_coord[i];
           birth_all[i][1] = Obverse_result->month_coord[i];
           birth_all[i][2] = Obverse_result->day_coord[i];
        }
        Obverse_result->birth_coord={
           *min_element(birth_all[0].begin(),birth_all[0].end()),
           *min_element(birth_all[1].begin(),birth_all[1].end()),
           *max_element(birth_all[2].begin(),birth_all[2].end()),
           *min_element(birth_all[3].begin(),birth_all[3].end()),
           *max_element(birth_all[4].begin(),birth_all[4].end()),
           *max_element(birth_all[5].begin(),birth_all[5].end()),
           *min_element(birth_all[6].begin(),birth_all[6].end()),
           *max_element(birth_all[7].begin(),birth_all[7].end())
           };
        
        //national                
        vector<int>index_class4;
        nclassindex(nclasses,index_class4,4);     
        if (index_class4.size()==1)
        {
            string class4_pattern = "[\u4e00-\u9fa5]+";
            string national_match = matchString(class4_pattern,reg_result[index_class4[0]]); 
            if (national_match=="穿青人"){
                Obverse_result->national = national_match; 
                Obverse_result->national_coord = bbox[index_class4[0]];               
            }
            else{
                Obverse_result->national = judge_nationality(national_match); 
                Obverse_result->national_coord = bbox[index_class4[0]];                 
            }
            
        } 
    
        //sex
        vector<int>index_class5;
        nclassindex(nclasses,index_class5,5);     
        if (index_class5.size()==1){
            string class5_pattern = "[\u4e00-\u9fa5]+";
            string sex_match = matchString(class5_pattern,reg_result[index_class5[0]]); 
            Obverse_result->sex = fix_sex(sex_match);
            Obverse_result->sex_coord = bbox[index_class5[0]];
        }  
        
        //address
        vector<int>index_class6;
        nclassindex(nclasses,index_class6,6);     
        if (index_class6.size()>=1)
        {
            vector<vector<float> > address_all(8,vector<float>(index_class6.size(),0));

            for(int i=0;i<address_all.size();i++)
            {
               for (int box_index=0;box_index<index_class6.size();box_index++)
               {
                   address_all[i][box_index] = bbox[index_class6[box_index]][i];
               }
            }
            auto index_ = sort_indexes(address_all[1]);
            string content = "";

            for(int i=0;i<index_.size();i++){
                content += reg_result[index_class6[index_[i]]];
            }
            string class6_pattern = "[—a-zA-Z0-9\u4e00-\u9fa5-.·]+"; //—
            Obverse_result->address = matchString(class6_pattern,content);
            Obverse_result->address_coord = {
                *min_element(address_all[0].begin(),address_all[0].end()),
                *min_element(address_all[1].begin(),address_all[1].end()),
                *max_element(address_all[2].begin(),address_all[2].end()),
                *min_element(address_all[3].begin(),address_all[3].end()),
                *max_element(address_all[4].begin(),address_all[4].end()),
                *max_element(address_all[5].begin(),address_all[5].end()),
                *min_element(address_all[6].begin(),address_all[6].end()),
                *max_element(address_all[7].begin(),address_all[7].end())
                };
        }                      
                    
        //id_number
        vector<int>index_class7;
        nclassindex(nclasses,index_class7,7);     
        if (index_class7.size()==1){
            string class7_pattern = "[0-9Xx]+";
            string id_number = matchString(class7_pattern,reg_result[index_class7[0]]);
            Obverse_result->id_number = id_number;
            Obverse_result->id_number_coord = bbox[index_class7[0]];
            //根据身份证号码的倒数第二位判断性别
            if (id_number.length()==18)
            {
                if (id_number[id_number.length()-2] %2 == 0)
                {
                   Obverse_result->sex = "女"; 
                }
                else
                {
                  Obverse_result->sex = "男";  
                }
            }
        } 
    }
    
}

string vecTostring(const vector<string> vec,size_t start,size_t end)
{
    string res="";
    for (size_t i=start;i<min(end,vec.size());i++){
        res += vec[i];
    }
    return res;   
}

bool search(const vector<string> strs, string target)
{
//     vector<string> ::iterator t;
    auto t = find(strs.begin(),strs.end(),target);
    if(t != strs.end()){
        return true;
    }
    return false;
}

void get_result_reserve(ID_Result_Reverse *Reverse_result,vector<int>&nclasses,vector<vector<float> >& bbox,vector<string>&reg_result)
{
    if (reg_result.size()>0)
    {
        //issued
        vector<int>index_class8;
        nclassindex(nclasses,index_class8,8);//取出标签为8的识别结果，是要是issued
//         cout<<"issued检测："<<index_class8.size()<<endl;
        if (index_class8.size()==1)
        {
            string class8_pattern = "[\u4e00-\u9fa5]+";//汉字
            string issued_match = matchString(class8_pattern,reg_result[index_class8[0]]);
            Reverse_result->issued = issued_match;
            vector<string>issued_match_s = split_utf8(issued_match);
            if (issued_match_s.size() > 2)
            {
                if (issued_match_s[issued_match_s.size()-2]=="分")
                {
                    issued_match_s[issued_match_s.size()-2] = "分";
                    issued_match_s[issued_match_s.size()-1] = "局";
                    Reverse_result->issued = vecTostring(issued_match_s,0,issued_match_s.size());
                }
            }
            Reverse_result->issued_coord = bbox[index_class8[0]];
        }   
        //effective_date
        vector<int>index_class9;
        nclassindex(nclasses,index_class9,9);  
        cout<<index_class9.size()<<endl;
        if (index_class9.size()==1)
        {
            string class9_pattern = "[0-9长期]+";
            string effective_date_match = matchString(class9_pattern,reg_result[index_class9[0]]); 
            vector<string>effective_date_match_s = split_utf8(effective_date_match);
            if (effective_date_match_s.size()==16)
            {
                Reverse_result->effective_date = 
                    vecTostring(effective_date_match_s,0,4)+"."
                    +vecTostring(effective_date_match_s,4,6)+"."
                    +vecTostring(effective_date_match_s,6,8)+"-"
                    +vecTostring(effective_date_match_s,8,12)+"."
                    +vecTostring(effective_date_match_s,12,14)+"."
                    +vecTostring(effective_date_match_s,14,effective_date_match_s.size());
            }
            else if (search(effective_date_match_s, "长") || search(effective_date_match_s, "期"))
            {
                effective_date_match = matchString("[0-9]+",effective_date_match);
                vector<string>effective_date_match_s = split_utf8(effective_date_match);
                if (effective_date_match_s.size()==8)
                {
                    Reverse_result->effective_date = 
                        vecTostring(effective_date_match_s,0,4)+"."
                        +vecTostring(effective_date_match_s,4,6)+"."
                        +vecTostring(effective_date_match_s,6,8)+"-"+"长期";                    
                }
            }
            else
            {
                Reverse_result->effective_date = effective_date_match;
            }
            Reverse_result->effective_date_coord = bbox[index_class9[0]];
        }  
//         cout<<"lalalalalalal"<<endl;
    }      
}


cv::Mat imresize(const cv::Mat &img, int img_max_size){
    int max_size = max(img.cols, img.rows);
    float ratio_re = (float) img_max_size / max_size;
    int w = (int) ((ratio_re * img.cols)/32) *  32;
    int h = (int) ((ratio_re * img.rows)/32) *  32;
    cv::Mat img2;
    cv::resize(img, img2, cv::Size(w, h));
//     cv::imwrite("/src/notebooks/ID_reg/hello_world/test_cpp.jpg",img2); 
    return img2;
}


cv::Mat SIFTInF(const cv::Mat &inpic_ori, const cv::Mat &template_img,float ratio,string img_type)
{
    cv::Mat inpic = imresize(inpic_ori,640);
    cv::Mat result;
    cv::Mat grayPic;
    cv::Mat graytemplate;
    cv::cvtColor(inpic, grayPic,cv::COLOR_BGR2GRAY);
    cv::cvtColor(template_img,graytemplate, cv::COLOR_BGR2GRAY);
    string obverse = "obverse";
    cv::Mat img1;
    cv::Mat img2;
    if (img_type==obverse){
        img1 = grayPic(cv::Range::all(),cv::Range(0,(int) grayPic.cols/2)).clone();
        img2 = graytemplate(cv::Range::all(),cv::Range(0,(int) graytemplate.cols*2/3)).clone();
    }
    else{
        img1 = grayPic.clone();
        img2 = template_img.clone();      
   } 
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2; 
    cv::Mat descriptors1, descriptors2;
    
    sift->detectAndCompute(img1,cv::Mat(), keypoints1, descriptors1);
    sift->detectAndCompute(img2,cv::Mat(), keypoints2, descriptors2); 
//     特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
//     KNN-NNDR匹配法
    std::vector<std::vector<cv::DMatch> > knn_matches;
    std::vector<cv::DMatch> good_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2);
    for (auto & knn_matche : knn_matches) {
        if (knn_matche[0].distance < ratio * knn_matche[1].distance) {
            good_matches.push_back(knn_matche[0]);
        }
    }
    if (good_matches.size() > 4)
    {
        vector<cv::Point2f> src_pts;
        vector<cv::Point2f> dst_pts;
        for (size_t i=0;i<good_matches.size();i++)
        {
            src_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
            dst_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);
        }
        cv::Mat H = cv::findHomography(src_pts,dst_pts,cv::RANSAC,10);
//         cout<<"H:"<<H<<endl;
//         cout<<"H:"<<H.cols<<endl;
        H.convertTo(H, CV_32F);
//         cout<<"H:"<<H.at<float>(0,0)<<endl;
//         cout<<"H:"<<H.at<float>(1,1)<<endl;
        
        if (H.cols > 0){
            if ((H.at<float>(0,0) > 0.5 && H.at<float>(0,0) < 2)&&(H.at<float>(1,1) > 0.5 && H.at<float>(1,1) < 2))
            {
                cv::warpPerspective(inpic, result, H, cv::Size(template_img.cols, template_img.rows));
                
            } 
        }
        
    }
    
    return result;
}


cv::Mat getPerImgInF(const cv::Mat &inpic,string img_type,bool &get_perspect)
{
    string obverse = "obverse";
    cv::Mat result;
    cv::Mat template_img;

    if (img_type==obverse){
        template_img = cv::imread("/src/notebooks/ImgIDRecogProj/utils/templateZ.jpg");
    }
    else{
        template_img = cv::imread("src/notebooks/ImgIDRecogProj/utils/templateF.jpg");
    }
    for( float ratio = 0.7; ratio < 1; ratio = ratio + 0.1)
    {
        try
        {
            result = SIFTInF(inpic, template_img, ratio,img_type);
        }
        catch(exception e){
            get_perspect = false;
            result = inpic.clone();
            return result;
        }
        
        if (result.cols>10)
        {
            if (img_type==obverse){
//                 cout<<JudgeImgPSuccessZ(template_img,result)<<endl;
                if (JudgeImgPSuccessZ(template_img,result)){
                    get_perspect = true;
                    return result;
                }
                else
                    continue;
            }
            else
            {
                if (JudgeImgPSuccessF(template_img,result))
                {
                    get_perspect = true;
                    return result;
                }
                else
                    continue;                
            }
            
        }
    }
    return inpic;
}
int main(int argc, char** argv)
{
    
    
    //批量测试
    ifstream inf;
    inf.open("/src/notebooks/IDtestData/TESTZLabel.txt");
    string s;
    int total_ID_Z=0;
    double total_time=0;
    while(getline(inf,s))
    {       
        
//         测试身份证反面数据
//         string img_path,label,issued,date;
//         int len_s=s.length();
//         int loc =s.find("	");
//         img_path="/src/notebooks/IDtestData/IDTEST1/"+s.substr(0,loc);
//         label=s.substr(loc+1,len_s-1);
//         issued=label.substr(0,label.find(','));
//         date=label.substr(label.find(',')+1,len_s-1);

        
//         测试身份证正面数据
        string img_path,ID_name,ID_birth,ID_national,ID_sex,ID_address,ID_id_number;
        int len_s=s.length();
        int loc=s.find("	");
        img_path="/src/notebooks/IDtestData/IDTEST0/"+s.substr(0,loc);
        string label=s.substr(loc+1,len_s-1);
        ID_name=label.substr(0,label.find(','));
        label=label.substr(label.find(',')+1,len_s-1);
        ID_birth=label.substr(0,label.find(','));
        label=label.substr(label.find(',')+1,len_s-1);  
        ID_national=label.substr(0,label.find(','));
        label=label.substr(label.find(',')+1,len_s-1);
        ID_sex=label.substr(0,label.find(','));
        label=label.substr(label.find(',')+1,len_s-1);
        ID_address=label.substr(0,label.find(','));
        label=label.substr(label.find(',')+1,len_s-1);
        ID_id_number=label.substr(0,label.find(','));
        label=label.substr(label.find(',')+1,len_s-1);
//         cout<<img_path<<" "<<ID_name<<" "<<ID_birth<<" "<<ID_national<<" "<<ID_sex<<" "<<ID_address<<" "<<ID_id_number<<endl;

    
    
//         cout<<img_path<<endl;
        cv::Mat m = cv::imread(img_path, 1); //BGR
//         cv::imwrite("../Output/ROI_sfz_ori.jpg",m);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", img_path.c_str());
            return -1;
        }

        std::vector<Object> objects;
        std::vector<Object> edgeCoords;
        std::vector<Object> textbbox;
        std::vector<vector<float>>textbboxes;
        std::vector<int> nclasses;
        string yolo_detect_model_path,yolo_detect_edge_model_path,yolo_detect_text_model_path,crnn_rec_text_model_path;

        yolo_detect_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/best_lite-sim_int8.param";
        yolo_detect_edge_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/best-edge-int8.param";
        yolo_detect_text_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/best-text-int8.param";
        crnn_rec_text_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/id_crnn_mobile_fix_sim_int8.param";
        struct timeb startTime_pro,endTime_pro;
        struct timeb startTime_sfz,endTime_sfz,startTime_edg,endTime_edg,startTime_text,endTime_text,startTime_rec,endTime_rec;
        ftime(&startTime_pro);
//         ftime(&startTime_sfz);
        try
        {
            detect_yolov5(m, yolo_detect_model_path, objects,0.50);//身份证目标检测
        
        }
        catch(exception e)
        {
            cout<<"输入图像： "<<img_path<<endl;
            cout<<"身份证目标检测存在错误"<<endl;
        
        }
        
//         ftime(&endTime_sfz);
//         cout<<"sfz检测时间"<<(endTime_sfz.time-startTime_sfz.time)*1000 + (endTime_sfz.millitm - startTime_sfz.millitm) <<"ms"<<endl;
    //     cout<<"num of boxes: "<<objects.size()<<endl;

        //画出目标框
    //     draw_objects(m, objects);


    //     //扣出目标图片
    //     int x_c,y_c,w_o,h_o; 
    //     for(int i=0;i<objects.size();i++)
    //     {
    //         const Object& obj = objects[i];
    //         x_c=(int)obj.rect.x;
    //         y_c=(int)obj.rect.y;
    //         w_o=(int)obj.rect.width;
    //         h_o=(int)obj.rect.height;
    //     }
    //     cout<<x_c<<endl;
    //     cout<<y_c<<endl;
    //     cout<<w_o<<endl;
    //     cout<<h_o<<endl;
    //     cv::Mat imgCrop;
    //     m(cv::Rect(x_c, y_c,w_o, h_o)).copyTo(imgCrop);
    //     cv::imwrite("./ROI_sfz.jpg",imgCrop);


        cv::Mat im_c,imshow,im_p,im_r;
        bool perspective;
        string ID_type;
        int flag=1;
    //     int a=1,b=2;
    //     cout<<"a+b="<<add(a,b)<<endl;
        for(int i=0;i<objects.size();i++) ///该for循环对应图像中的sfz图片
        {
            std::vector<float> result_conf;
            vector<string>recog_result;
            const Object& obj = objects[i];
            GetCropImageAndIDType(im_c,ID_type,m,obj,true);//截取目标的ROI得到目标im_c,并且返回当前类别
            im_c.copyTo(imshow);
//             ftime(&startTime_edg);
            try
            {
                detect_yolov5(im_c, yolo_detect_edge_model_path, edgeCoords,0.25);//边缘检测
            }
            catch(exception e)
            {
                cout<<"输入图像： "<<img_path<<endl;
                cout<<"边缘检测存在error"<<endl;
            
            }
//            ftime(&endTime_edg);
//            cout<<"edg检测时间"<<(endTime_edg.time-startTime_edg.time)*1000 + (endTime_edg.millitm - startTime_edg.millitm) <<"ms"<<endl;
            
//             cout<<"num of edge_boxes: "<<edgeCoords.size()<<endl;
//             draw_objects(im_c, edgeCoords);
//             cv::imwrite("../Output/img_edg.jpg",im_c);

            if(edgeCoords.size()>0)
            {
                GetImgByEdge(im_p,ID_type,im_c,edgeCoords,perspective);//边缘检测后进行校验
    //             cout<<"perspective"<<perspective<<endl;
    //             cout<<"im_p形状："<<im_p.size()<<endl;
    //             cout<<"im_r形状："<<im_r.size()<<endl;
    //             cout<<"im_c形状："<<im_c.size()<<endl;
                if(perspective && !(im_p.empty()))
                {
                    im_p.copyTo(im_r);
//                     cv::imwrite("../Output/im_r.jpg",im_r);
                }
                else
                {

//                     cout<<"进行sift变换"<<endl;
                    GetCropImageAndIDType(im_p,ID_type,m,obj,false);

    //                 SIFT变换
                    bool get_perspect = false;
                    im_r=getPerImgInF(im_p,ID_type,get_perspect);  
//                     cv::imwrite("../Output/sift_im_r.jpg",im_r);

                }
                //画出检测到的edge的object
                for(int i=0;i<edgeCoords.size();i++)
                {
                   const Object& edge_obj = edgeCoords[i];
                   cv::rectangle(imshow, edge_obj.rect, cv::Scalar(255, 0, 0));
//                    cv::imwrite("../Output/edge_ob.jpg",imshow);
                }     
            }
            else
            {
                  GetCropImageAndIDType(im_r,ID_type,m,obj,false);
                    //SIFT变换
                   bool get_perspect = false;
                   im_r=getPerImgInF(im_r,ID_type,get_perspect); 
//                    cv::imwrite("../Output/sift_im_r.jpg",im_r);
            }


            ////////////////Text detect//////////////////
//            cv::imwrite("../Output/im_r.jpg",im_r);
//            cv::imwrite("../Output/im_p.jpg",im_p);
//            ftime(&startTime_text);
           try
           {
               detect_yolov5(im_r, yolo_detect_text_model_path, textbbox,0.50);
//                for(int i=0;i<textbbox.size();i++)
//                { 
//                     const Object& obj = textbbox[i];
//                     cout<<obj.label<<" "<<obj.prob<<endl;
//                }
               
           }
           catch(exception e)
           {
               cout<<"输入图像： "<<img_path<<endl;
               cout<<"文本检测存在error"<<endl;
           
           }
           
//            ftime(&endTime_text);
//            cout<<"text检测时间"<<(endTime_text.time-startTime_text.time)*1000 + (endTime_text.millitm - startTime_text.millitm) <<"ms"<<endl;
    //        cout<<"num of text_boxes:"<<textbbox.size()<<endl;
//            draw_objects(im_r, textbbox);


           fixbox(textbboxes,nclasses,textbbox); //将xywh转换成x1y1x2y1x2y2x1y2格式
//            cout<<textbboxes.size()<<endl;
//            cout<<nclasses.size()<<endl;
           //text log添加,'Detect IDCard Text OK'

            //打印出来看看textbbox
//            for(int i=0;i<(int)(textbboxes.size());i++)
//            {
//                for(int j=0;j<(int)(textbboxes[i].size());j++)
//                {
//                    cout<<textbboxes[i][j]<<" ";
//                }
//                cout<<endl;

//            }

            //扣出目标图片
    //         int x_c,y_c,w_o,h_o;
           string result_rec;
//            ftime(&startTime_rec);
           for(int i=0;i<textbboxes.size();i++)  //遍历检测得bbox，扣除ROI并且识别
           {
               cv::Mat img_c_text,im_c_text_p;
               vector<float>coord;
               coord.push_back(textbboxes[i][0]);
               coord.push_back(textbboxes[i][1]);
               coord.push_back(textbboxes[i][2]);
               coord.push_back(textbboxes[i][3]);
               coord.push_back(textbboxes[i][4]);
               coord.push_back(textbboxes[i][5]);
               coord.push_back(textbboxes[i][6]);
               coord.push_back(textbboxes[i][7]);
    //         im_r(cv::Rect(x_1, y_1,w_r, h_r)).copyTo(img_c_text); //传入左上角的点和宽高或者使用下面的函数进行扣取
              cv::Mat coord_point=cv::Mat(coord,true); //8*1
              im_c_text_p=get_perspective_image_rec_result(im_r,coord_point); ///图像扣取校正
            //在c++张遍历保留不同名字的方式
//               std::ostringstream name;
//               name<<"text_detect_pers_"<<i<<".jpg";
//               cv::imwrite("/src/notebooks/IDtestData/text_de_pers/"+name.str(),im_c_text_p);

              ///////////////crnn////////////////////////
              //对每一个text_detect ROI进行识别
             try
              {    
                  result_rec=detect_crnn(im_c_text_p,crnn_rec_text_model_path,result_conf);
//                   cout<<"识别结果："<<result_rec<<" ";
              }
             catch(exception e)
             {
                   cout<<"输入图像： "<<img_path<<endl;
                   cout<<"文本识别存在error"<<endl;
                   
             }
              
    //         cout<<result_rec<<endl;

              recog_result.push_back(result_rec);  
    //         break;    

          }
//             for(int i=0;i<recog_result.size();i++)
//             {
//                 cout<<recog_result[i]<<endl;
//             }
            
//          ftime(&endTime_rec);
//          cout<<"text识别时间"<<(endTime_rec.time-startTime_rec.time)*1000 + (endTime_rec.millitm - startTime_rec.millitm) <<"ms"<<endl;
          //后处理
            if((int)(textbboxes.size())<6 && ID_type=="obverse")
            {
                
                string re_img=img_path;
                string img_save_name="/src/notebooks/IDtestData/c++_reverse/"+re_img.substr(re_img.rfind("/")+1,re_img.length()); 
                cout<<img_save_name<<endl;
//                 cv::imwrite(img_save_name,m);
                cout<<"类别出错，应该属于背面"<<endl;
                ID_type="reverse";
            }
            else if((int)(textbboxes.size())>6 && ID_type=="reverse")
            {
                string re_img=img_path;
                string img_save_name="/src/notebooks/IDtestData/c++_obverse/"+re_img.substr(re_img.rfind("/")+1,re_img.length()); 
                cout<<img_save_name<<endl;
//                 cv::imwrite(img_save_name,m);
                cout<<"类别出错，应该属于正面"<<endl;
                ID_type="obverse";
            }
            struct ID_Result_Obverse Obverse_result; 
            struct ID_Result_Reverse Reverse_result;
            if (ID_type=="obverse")
            {
                get_result_observe(&Obverse_result,nclasses,textbboxes,recog_result);  
             
//                 cout<<"姓名：" <<Obverse_result.name <<" birth："<<Obverse_result.birth <<" address："<<Obverse_result.address <<endl;
//                 cout<<"year：" <<Obverse_result.year <<" month："<<Obverse_result.month <<" day："<<Obverse_result.day <<endl;
//                 cout<<"national：" <<Obverse_result.national <<" sex："<<Obverse_result.sex <<" id_number："<<Obverse_result.id_number <<endl;
            }
            else
            {
                get_result_reserve(&Reverse_result,nclasses,textbboxes,recog_result);
//                 cout<<"issued coord：" <<Reverse_result.issued_coord[1]<<endl;
//                 cout<<"issued：" <<Reverse_result.issued<<endl;
//                 cout<<"effective_date：" <<Reverse_result.effective_date <<endl;
//                 cout<<"effective_date coord：" <<Reverse_result.effective_date_coord[1] <<endl;
            }
            //log 'Get IDCard Item OK'
            
            typedef map<string, string> Recog_Item; 
            Recog_Item recog_item; 
            if(ID_type=="obverse")
            {
                recog_item["name"]=Obverse_result.name;
                recog_item["sex"]=Obverse_result.sex;
                recog_item["national"]=Obverse_result.national;
                recog_item["birth"]=Obverse_result.birth;
                recog_item["address"]=Obverse_result.address;
                recog_item["id_number"]=Obverse_result.id_number;
                recog_item["id_type"]=ID_type;
                
                if((recog_item["name"]==ID_name) && (recog_item["sex"]==ID_sex) && (recog_item["national"]==ID_national)
                  && (recog_item["birth"]==ID_birth) && (recog_item["address"]==ID_address) && (recog_item["id_number"]==ID_id_number))
                {
//                     cout<<img_path<<"识别正确"<<endl;
                    total_ID_Z+=1;

                }
                else
                {
//                     string img_name_no_path=img_path.substr(img_path.rfind("/"),img_path.length());
//                     string new_img_error_path="/src/notebooks/IDtestData/c++_img_IDRECO_error/"+img_name_no_path;
//                     cv::imwrite(new_img_error_path,m);
                    cout<<"真实标签："<<ID_name<<" "<<ID_sex<<" "<<ID_national<<" "<<ID_birth<<" "<<ID_address<<" "<<ID_id_number<<endl;
                    cout<<"预测标签：";
                    cout<<img_path<<" "<<recog_item["name"]<<" "<<recog_item["sex"]<<" "<<recog_item["national"]<<" ";
                    cout<<recog_item["birth"]<<" "<<recog_item["address"]<<" "<<recog_item["id_number"]<<endl;
                    
                
                }
      
            }
            else
            {
                recog_item["issued"]=Reverse_result.issued;
                recog_item["effective_date"]=Reverse_result.effective_date;
                recog_item["id_type"]=ID_type;
                
//                 if((recog_item["issued"]==issued) && (recog_item["effective_date"]==date))
//                 {
//                     cout<<img_path<<"识别正确"<<endl;
//                     total_ID_Z+=1;

//                 }
//                 else
//                 {
//                     string img_name_no_path=img_path.substr(img_path.rfind("/"),img_path.length());
//                     string new_img_error_path="/src/notebooks/IDtestData/c++_img_IDRECO_error/"+img_name_no_path;
// //                     cv::imwrite(new_img_error_path,m);
//                     cout<<"真实标签："<<issued<<" "<<date<<endl;
//                     cout<<"预测标签：";
//                     cout<<img_path<<" "<<recog_item["issued"]<<" "<<recog_item["effective_date"]<<" "<<endl;
//                 }
                
            }
            flag+=1;

        }
        ftime(&endTime_pro);
        total_time+=(endTime_pro.time-startTime_pro.time)*1000 + (endTime_pro.millitm - startTime_pro.millitm); 
//         cout<<"整个项目的响应时间"<<(endTime_pro.time-startTime_pro.time)*1000 + (endTime_pro.millitm - startTime_pro.millitm) <<"ms"<<endl;
//         stop=clock();
//         double time_response=((double)(stop-start))/CLOCKS_PER_SEC; 
        

        
        
//         cout<<"respones time: "<<time_response<<"s"<<endl;
//         cv::imwrite("../Output/ROI_sfz111.jpg",im_c);
//         cout<<ID_type<<endl;
    }
//     cout<<total_time/3825.0<<endl;
    cout<<total_ID_Z<<endl;
    return 0;

/**************************************************/
//     //单独测试
//         if (argc != 2)
//         {
//             fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
//             return -1;
//         }
//         const char* img_path = argv[1];
//         cv::Mat m = cv::imread(img_path, 1); //BGR
// //         cv::imwrite("../Output/ROI_sfz_ori.jpg",m);
//         if (m.empty())
//         {
//             fprintf(stderr, "cv::imread %s failed\n", img_path);
//             return -1;
//         }

//         std::vector<Object> objects;
//         std::vector<Object> edgeCoords;
//         std::vector<Object> textbbox;
//         std::vector<vector<float>>textbboxes;
//         std::vector<int> nclasses;
//         string yolo_detect_model_path,yolo_detect_edge_model_path,yolo_detect_text_model_path,crnn_rec_text_model_path;

//         yolo_detect_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/best_lite-sim_int8.param";
//         yolo_detect_edge_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/best-edge-int8.param";
//         yolo_detect_text_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/best-text-int8.param";
//         crnn_rec_text_model_path="/src/notebooks/ncnn-20220420/build/tools/quantize/id_crnn_mobile_fix_sim_int8.param";
//         struct timeb startTime_pro,endTime_pro;
//         ftime(&startTime_pro);
// //         ftime(&startTime_sfz);
//         try
//         {
//             detect_yolov5(m, yolo_detect_model_path, objects,0.25);//身份证目标检测
//             cout<<"身份证检测成功"<<endl;
        
//         }
//         catch(exception e)
//         {
//             cout<<"身份证目标检测存在错误"<<endl;
        
//         }
        
// //         ftime(&endTime_sfz);
// //         cout<<"sfz检测时间"<<(endTime_sfz.time-startTime_sfz.time)*1000 + (endTime_sfz.millitm - startTime_sfz.millitm) <<"ms"<<endl;
//         cout<<"num of boxes: "<<objects.size()<<endl;

//         //画出目标框
//         draw_objects(m, objects);


//     //     //扣出目标图片
//         int x_c,y_c,w_o,h_o; 
//         for(int i=0;i<objects.size();i++)
//         {
//             const Object& obj = objects[i];
//             cout<<i<<"图置信度"<<obj.prob<<endl;
//             x_c=(int)obj.rect.x;
//             y_c=(int)obj.rect.y;
//             w_o=(int)obj.rect.width;
//             h_o=(int)obj.rect.height;
//             cv::Mat imgCrop;
//             m(cv::Rect(x_c, y_c,w_o, h_o)).copyTo(imgCrop);
//             string roi_name="/src/notebooks/IDtestData/sfz_ROI/ROI_sfz_"+to_string(i)+".jpg";
//             cout<<roi_name<<endl;
//             cv::imwrite(roi_name,imgCrop);
//         }
       

//         cv::Mat im_c,imshow,im_p,im_r;
//         bool perspective;
//         string ID_type;
//         int flag=1;
//     //     int a=1,b=2;
//     //     cout<<"a+b="<<add(a,b)<<endl;
//         cout<<"检测的身份证数目: "<<objects.size()<<endl;  
//         for(int i=0;i<objects.size();i++) ///该for循环对应图像中的sfz图片
//         {
//             std::vector<float> result_conf;
//             vector<string>recog_result;
//             const Object& obj = objects[i];
//             GetCropImageAndIDType(im_c,ID_type,m,obj,true);//截取目标的ROI得到目标im_c,并且返回当前类别
//             im_c.copyTo(imshow);
// //             ftime(&startTime_edg);
//             try
//             {
//                 detect_yolov5(im_c, yolo_detect_edge_model_path, edgeCoords,0.25);//边缘检测
//                 cout<<"边缘检测成功"<<endl;
//             }
//             catch(exception e)
//             {
//                 cout<<"边缘检测存在error"<<endl;
            
//             }
// //            ftime(&endTime_edg);
// //            cout<<"edg检测时间"<<(endTime_edg.time-startTime_edg.time)*1000 + (endTime_edg.millitm - startTime_edg.millitm) <<"ms"<<endl;
            
// //             cout<<"num of edge_boxes: "<<edgeCoords.size()<<endl;
// //             draw_objects(im_c, edgeCoords);
// //             cv::imwrite("../Output/img_edg.jpg",im_c);

//             if(edgeCoords.size()>0)
//             {
//                 GetImgByEdge(im_p,ID_type,im_c,edgeCoords,perspective);//边缘检测后进行校验
//     //             cout<<"perspective"<<perspective<<endl;
//     //             cout<<"im_p形状："<<im_p.size()<<endl;
//     //             cout<<"im_r形状："<<im_r.size()<<endl;
//     //             cout<<"im_c形状："<<im_c.size()<<endl;
//                 if(perspective && !(im_p.empty()))
//                 {
//                     im_p.copyTo(im_r);
//                     cout<<"保存边缘检测后仿射变换的图像"<<endl;
//                     cv::imwrite("../Output/im_p.jpg",im_r);
//                 }
//                 else
//                 {

//                     cout<<"进行sift变换"<<endl;
//                     GetCropImageAndIDType(im_p,ID_type,m,obj,false);

//     //                 SIFT变换
//                     bool get_perspect = false;
//                     im_r=getPerImgInF(im_p,ID_type,get_perspect);  
//                     cv::imwrite("../Output/sift_im_r.jpg",im_r);

//                 }
//                 //画出检测到的edge的object
//                 for(int i=0;i<edgeCoords.size();i++)
//                 {
//                    const Object& edge_obj = edgeCoords[i];
//                    cv::rectangle(imshow, edge_obj.rect, cv::Scalar(255, 0, 0));
// //                    cv::imwrite("../Output/edge_ob.jpg",imshow);
//                 }     
//             }
//             else
//             {
//                   GetCropImageAndIDType(im_r,ID_type,m,obj,false);
//                     //SIFT变换
//                    bool get_perspect = false;
//                    im_r=getPerImgInF(im_r,ID_type,get_perspect); 
// //                    cv::imwrite("../Output/sift_im_r.jpg",im_r);
//             }


//             ////////////////Text detect//////////////////
// //            cv::imwrite("../Output/im_r.jpg",im_r);
// //            cv::imwrite("../Output/im_p.jpg",im_p);
// //            ftime(&startTime_text);
//            try
//            {
//                detect_yolov5(im_r, yolo_detect_text_model_path, textbbox,0.25);
//                cout<<"文本检测成功"<<endl;
//            }
//            catch(exception e)
//            {
//                cout<<"文本检测存在error"<<endl;
           
//            }
           
// //            ftime(&endTime_text);
// //            cout<<"text检测时间"<<(endTime_text.time-startTime_text.time)*1000 + (endTime_text.millitm - startTime_text.millitm) <<"ms"<<endl;
//     //        cout<<"num of text_boxes:"<<textbbox.size()<<endl;
//            draw_objects(im_r, textbbox);


//            fixbox(textbboxes,nclasses,textbbox); //将xywh转换成x1y1x2y1x2y2x1y2格式
// //            cout<<textbboxes.size()<<endl;
//            cout<<"检测到的text的类别（0~9）"<<nclasses.size()<<endl;
//            for(int i=0;i<nclasses.size();i++)
//            {
           
//            cout<<nclasses[i]<<endl;
//            }
//            //text log添加,'Detect IDCard Text OK'

//             //打印出来看看textbbox
// //            for(int i=0;i<(int)(textbboxes.size());i++)
// //            {
// //                for(int j=0;j<(int)(textbboxes[i].size());j++)
// //                {
// //                    cout<<textbboxes[i][j]<<" ";
// //                }
// //                cout<<endl;

// //            }

//             //扣出目标图片
//     //         int x_c,y_c,w_o,h_o;
//            string result_rec;
// //            ftime(&startTime_rec);
//            for(int i=0;i<textbboxes.size();i++)  //遍历检测得bbox，扣除ROI并且识别
//            {
//                cv::Mat img_c_text,im_c_text_p;
//                vector<float>coord;
//                coord.push_back(textbboxes[i][0]);
//                coord.push_back(textbboxes[i][1]);
//                coord.push_back(textbboxes[i][2]);
//                coord.push_back(textbboxes[i][3]);
//                coord.push_back(textbboxes[i][4]);
//                coord.push_back(textbboxes[i][5]);
//                coord.push_back(textbboxes[i][6]);
//                coord.push_back(textbboxes[i][7]);
//     //         im_r(cv::Rect(x_1, y_1,w_r, h_r)).copyTo(img_c_text); //传入左上角的点和宽高或者使用下面的函数进行扣取
//               cv::Mat coord_point=cv::Mat(coord,true); //8*1
//               im_c_text_p=get_perspective_image_rec_result(im_r,coord_point); ///图像扣取校正
// //             //在c++张遍历保留不同名字的方式
// //               std::ostringstream name;
// //               name<<"text_detect_pers_"<<i<<".jpg";
// //               cv::imwrite(name.str(),im_c_text_p);

//               ///////////////crnn////////////////////////
//               //对每一个text_detect ROI进行识别
//              try
//               {    
//                  result_rec=detect_crnn(im_c_text_p,crnn_rec_text_model_path,result_conf);
//                  cout<<"文本识别成功"<<endl;
//               }
//              catch(exception e)
//              {
 
//                    cout<<"文本识别存在error"<<endl;
                   
//              }
              
//     //         cout<<result_rec<<endl;

//               recog_result.push_back(result_rec);  
//     //         break;    

//           }
//            cout<<"textbboxes数量:"<<textbboxes.size()<<endl;
// //          ftime(&endTime_rec);
// //          cout<<"text识别时间"<<(endTime_rec.time-startTime_rec.time)*1000 + (endTime_rec.millitm - startTime_rec.millitm) <<"ms"<<endl;
//           //后处理
//             if((int)(textbboxes.size())<6 && ID_type=="obverse")
//             {
//                 cout<<"类别出错，应该属于背面"<<endl;
//                 ID_type="reverse";
//             }
//             else if((int)(textbboxes.size())>6 && ID_type=="reverse")
//             {
//                 cout<<"类别出错，应该属于正面"<<endl;
//                 ID_type="obverse";
//             }
//             struct ID_Result_Obverse Obverse_result; 
//             struct ID_Result_Reverse Reverse_result;
//             if (ID_type=="obverse")
//             {
//                 get_result_observe(&Obverse_result,nclasses,textbboxes,recog_result);  
//                 cout<<"姓名：" <<Obverse_result.name <<" birth："<<Obverse_result.birth <<" address："<<Obverse_result.address <<endl;
//                 cout<<"year：" <<Obverse_result.year <<" month："<<Obverse_result.month <<" day："<<Obverse_result.day <<endl;
//                 cout<<"national：" <<Obverse_result.national <<" sex："<<Obverse_result.sex <<" id_number："<<Obverse_result.id_number <<endl;
//             }
//             else
//             {
//                 cout<<ID_type<<endl;
//                 cout<<recog_result[0]<<" "<<recog_result[1]<<endl;
//                 get_result_reserve(&Reverse_result,nclasses,textbboxes,recog_result);
//                 cout<<"issued coord：" <<Reverse_result.issued_coord[1]<<endl;
//                 cout<<"issued：" <<Reverse_result.issued<<endl;
//                 cout<<"effective_date：" <<Reverse_result.effective_date <<endl;
//                 cout<<"effective_date coord：" <<Reverse_result.effective_date_coord[1] <<endl;
//             }
//             //log 'Get IDCard Item OK'

//             typedef map<string, string> Recog_Item; 
//             Recog_Item recog_item; 
//             if(ID_type=="obverse")

//             {
//                 recog_item["name"]=Obverse_result.name;
//                 recog_item["sex"]=Obverse_result.sex;
//                 recog_item["national"]=Obverse_result.national;
//                 recog_item["birth"]=Obverse_result.birth;
//                 recog_item["address"]=Obverse_result.address;
//                 recog_item["id_number"]=Obverse_result.id_number;
//                 recog_item["id_type"]=ID_type;
                
      
//             }
//             else
//             {
//                 recog_item["issued"]=Reverse_result.issued;
//                 recog_item["effective_date"]=Reverse_result.effective_date;
//                 recog_item["id_type"]=ID_type;
//             }
//             flag+=1;

//         }
//         ftime(&endTime_pro);
//         cout<<"整个项目的响应时间"<<(endTime_pro.time-startTime_pro.time)*1000 + (endTime_pro.millitm - startTime_pro.millitm) <<"ms"<<endl;
// //         stop=clock();
// //         double time_response=((double)(stop-start))/CLOCKS_PER_SEC; 
        

        
        
// //         cout<<"respones time: "<<time_response<<"s"<<endl;
// //         cv::imwrite("../Output/ROI_sfz111.jpg",im_c);

// //         cout<<ID_type<<endl;

//         return 0;

}
