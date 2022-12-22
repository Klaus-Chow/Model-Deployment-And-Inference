// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "net.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <iostream>
#include <exception>
#include <time.h>
using namespace std;



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


    for (int i = 0; i < h; i++) {
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
//             cout<<maxIndex<<endl;
            strRes.append(keys[maxIndex - 1]);
        }
        lastIndex = maxIndex;
        
    }
    return strRes;
}
static std::string detect_crnn(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    std::string Res;
    ncnn::Net crnn;
    crnn.opt.num_threads=8;
    crnn.opt.use_int8_inference=true;

//     crnn.opt.use_vulkan_compute = true;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    
    //模型1：crnn_moblie_op11_sim_fp16.param，crnn_moblie_op11_sim_fp16.bin
    //模型2：crnn_moblie_mobilev3-augTrue_sim_fp16
    //模型3：BaiDu_netCRNN_fp16
    
    crnn.load_param("/src/notebooks/ncnn-20220420/build/tools/id_crnn_mobile_fix_sim_fp32.param");
    crnn.load_model("/src/notebooks/ncnn-20220420/build/tools/id_crnn_mobile_fix_sim_fp32.bin");

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

//单独测试
// int main(int argc, char** argv)
// {
//     string result_rec,image_path;
//     image_path="/src/notebooks/ncnn-20220420/build/text_ROI_sfz.jpg";
// //         cout<<image_path<<endl;
// //         cout<<label<<endl;
//     cv::Mat m = cv::imread(image_path, 1);
//     std::vector<float> cls_scores;
//     result_rec=detect_crnn(m,cls_scores);
//     cout<<result_rec.length()<<endl;
//     cout<<result_rec<<endl;
        
//     return 0;
// }



// 批量测试
int main(int argc, char** argv)
{
    
    //加载label
    clock_t start,end;
    int acc=0,total=0;
    double time_response=0;
    string s,image_path,label;
    ifstream inf;
    string result_rec;
    inf.open("/src/notebooks/MyWorkData/IDNEWDATA20220624Train/new_train.txt");
    while(getline(inf,s))
    {
    
        total+=1;
        int len_s=s.length();
//         cout<<s<<endl;
//         cout<<len_s<<endl;
        //获取图像名称
        int local1=s.find('\t'); //从左侧开始找到一个空格返回index
        string img_path=s.substr(0,local1);
//         cout<<local1<<endl;
//         cout<<img_path<<endl;
        
        int local2=s.rfind('\t');
        string label=s.substr(local2+1,len_s-1);
//         cout<<local2<<endl;
//         cout<<label<<endl;
        
        cv::Mat m = cv::imread(img_path, 1);//bgr

        std::vector<float> cls_scores;
        try{
            start=clock();
            result_rec=detect_crnn(m,cls_scores);
            end=clock();
            time_response+=((double)(end-start))/CLOCKS_PER_SEC; 
        }
        catch(exception &e){
            cout<<img_path<<endl;
        }
        //         cout<<result_rec.length()<<endl;
        
        if(label==result_rec)
        {
            acc+=1;
        }
        else
        {
        cout<<"识别结果"<<" "<<result_rec<<"  真实标签:"<<label<<endl;
        }  
    }
    cout<<acc<<endl;
    cout<<total<<endl;
    cout<<(float)(acc*1.0/total*1.0)<<endl;
    cout<<time_response/total<<endl;
//     inf.close();
    return 0;
}
