#include<iostream>
#include "yolov5_virtual.h"
#include "Utils_yolo.h"
using namespace std;
void YoloInfer:: detect(const cv::Mat& bgr,vector<Object>& objects)
{    
    /*
        bgr:带检测的图像（bgr）
        objects:vector<Object> objects;用于返回检测出的bbox
        pro_threshold:分类阈值
    
    */
    const int target_size = 640;

    int img_w = bgr.cols;
    int img_h = bgr.rows;
//     cout<<"输入图像的尺寸"<<img_h<<" "<<img_w<<endl;
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
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    
    
    int wpad = (w + 32 - 1) / 32 * 32 - w;
    int hpad = (h + 32 - 1) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

//     cout<<"padding后的图像尺寸"<<in_pad.h<<" "<<in_pad.w<<endl;
    
    
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;
    
    
     // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output", out);
//         cout<<"stride8 输出"<<out.h<<" "<<out.w<<" "<<out.c<<endl;//(h/8)*(w/8),(8+5),3
        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);//在grid上得到所有的预测的bbox(并且根据识别阈值删除一部分阈值低的bboxes)

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());//不断从尾部插入object
    }

    // stride 16
    {
        ncnn::Mat out;

        ex.extract("417", out);
//         cout<<"stride16 输出"<<out.h<<" "<<out.w<<" "<<out.c<<endl;//(h/16)*(w/16),(8+5),3
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

        ex.extract("437", out);
//         cout<<"stride32 输出"<<out.h<<" "<<out.w<<" "<<out.c<<endl;//(h/32)*(w/32),(8+5),3
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

    
    //nms部分
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
//     cout<<"所有的bbox数量"<<count<<endl;
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded,减去padding,得到原图像（未padding）的坐标
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
}    