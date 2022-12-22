#include "Utils_yolo.h"
#include<iostream>
using namespace std;

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
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

void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
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
        const Object& a = faceobjects[i];//当前的proposal与picked中每一个bbox进行IOU计算，如果小于阈值则入栈

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)//0.45
                keep = 0;//不入栈
        }

        if (keep)
            picked.push_back(i);
    }
}
void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    /*
    anchors:(1,6)
    in_pad:(in_pad.h,in_pad.w)
    feat_blob:(in_pad.h/stride)*(in_pad.w/stride),numclass+5,num_anchors)
    */
    const int num_grid = feat_blob.h; //(in_pad.h/stride)*(in_pad.w/stride)

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
//     cout<<"num_grid_y: "<<num_grid_y<<endl;
//     cout<<"num_grid_x: "<<num_grid_x<<endl;
    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;//三种anchor:3

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];//(anchors[0],anchor[1]),(anchor[2],anchor[3]),(anchor[4],anchor[5])
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j); //取出
            

                // find class index with max class score
                //featptr (tx,ty,tw,th,obj,cls1,cls2,...)
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
                }//找出最有可能类别

                float box_score = featptr[4];
                
                //条件概率：当前grid存在某个类别的概率=当前grid存在目标的概率*最有可能类别的概率
                //这里使用sigmoid函数， sigmoid(box_score)表示是否存在obj,sigmoid(class_score)表示是否为该类别
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

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;//还原到原图像*stride，注意这里只是还原到in_pad的图像上
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;
                    
                    //根据偏移量转换成左上和右下角点
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
void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects,string img_name)
{
    static const char* class_names[] = {
        "mtz"
//         "name","born_y","born_m","born_d","national","gender","location","ID","local","time","face"
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
    
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        
//         cout<<obj.label<<endl;
//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));
        
        int classes_index=obj.label;
        const string text=class_names[classes_index];
//         sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

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
    string img_save_name="../Output/detect_"+img_name.substr(img_name.rfind("/")+1,img_name.length());
    cv::imwrite(img_save_name,image);
//     cv::imshow("image", image);
   
}
void crop_ROI(const cv::Mat& bgr,cv::Mat& ROI,const std::vector<Object>& objects,string img_name)
{
    for(int i=0;i<objects.size();i++)
    {
        const Object& obj=objects[i];
        int x_c,y_c,w,h;
        x_c=(int)obj.rect.x;
        y_c=(int)obj.rect.y;
        w=(int)obj.rect.width;
        h=(int)obj.rect.height;
        string img_save_name="../Output/ROI_"+to_string(i)+"_"+img_name.substr(img_name.rfind("/")+1,img_name.length());
        bgr(cv::Rect(x_c,y_c,w,h)).copyTo(ROI);
        cv::imwrite(img_save_name,ROI);
    
    }
}