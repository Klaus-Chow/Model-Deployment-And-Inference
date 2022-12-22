#ifndef CRNN_UTILS
#define CRNN_UTILS
#include <string>
#include <vector>
using namespace std;
std::string scoreToTextLine(const float *outputData, int h, int w);
std::vector<std::string> split_chinese(std::string s);
#endif