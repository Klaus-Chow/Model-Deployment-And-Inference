#ifndef NET_DETECT_H
#define NET_DETECT_H
#include "Net_Creator.h"
void Net_detect(Net_Creator *n,cv::Mat &bgr,string &rec)
{

    n->detect(bgr,rec);


}
void Net_detect(Net_Creator *n,cv::Mat &bgr,vector<Object>& objects)
{

    n->detect(bgr,objects);
}
#endif