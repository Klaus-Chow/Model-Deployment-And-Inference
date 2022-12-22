#include "Utils_crnn.h"
#include<string>
#include<fstream>
#include <algorithm>
#include<cmath>
#include <numeric>
using namespace std;

//从txt文本读取中文汉字(用ifstream读取txt文件后得到string类型，再用该函数切出每个汉字)
std::vector<std::string> split_chinese(std::string s) 
{
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
//根据分数排序，返回对应index的文字
string scoreToTextLine(const float *outputData, int h, int w)
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