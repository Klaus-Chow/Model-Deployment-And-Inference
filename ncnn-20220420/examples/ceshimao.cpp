// // #include <iostream>
// // #include <string>
// // #include <vector>
// // #include <map>

// // using namespace std;
// // int main()
// // {
// //    typedef map<char, vector<vector<int>>> Eclass;
// //    vector<vector<int>>v,v99;
// //    vector<int> v1;
// //    vector<int> v2;
// //    vector<int> v3;
// //    vector<int> v4;
// //    for (int i = 0;i < 5;i++)
// // 	{
// //        v1.push_back(i + 1);
// //        v2.push_back(i + 2);
// //        v3.push_back(i + 3);
// //        v4.push_back(i + 4);
// // 	}
// //    v.push_back(v1);
// // 	v.push_back(v2);
// // 	v.push_back(v3);
// //    v99.push_back(v4);
   
// // //    for (vector<vector<int>>::iterator it = v.begin();it != v.end();it++)//通过迭代器访问数据
// // // 	{
// // // 		//*it 代表的是vector<int>，又是一个容器，所以需要把小容器里边的数据遍历出来
// // // 		//小技巧：*it 就是vector 尖括号 里边的数据类型

// // // 		for (vector<int>::iterator vit = (*it).begin();vit != (*it).end();vit++)
// // // 		{
// // // 			cout << *vit <<" ";
// // // 		}
// // // 		cout << endl;   		
// // // 	}
   
// //    Eclass eclass; 
// //    eclass['0']=v;
// //    eclass['1']=v99;
// //    for(Eclass::iterator it=eclass.begin();it!=eclass.end();it++)
// //    {
// //        cout << "key:"<<it->first;
// //        for (vector<vector<int>>::iterator itv = (it->second).begin();itv != (it->second).end();itv++)//通过迭代器访问数据
// //         {
// // 		//*it 代表的是vector<int>，又是一个容器，所以需要把小容器里边的数据遍历出来
// // 		//小技巧：*it 就是vector 尖括号 里边的数据类型

// //             for (vector<int>::iterator vit = (*itv).begin();vit != (*itv).end();vit++)
// //             {
// //                 cout<<":"<<*vit <<" ";
// //             }
// //         }
// //        cout<<endl;
       
// //    }
    
// //     cout<< eclass.size()<<endl;
    
    
    
    
// //    return 0;
// // }
// // #include<iostream>
// // #include<map>
// // #include<vector>
// // using namespace std;
// // int main()
// // {
// //      typedef map<char,vector<vector<int>>> Eclass;
// //      Eclass eclass;
// //      vector<int> xy_c,z;
// //      xy_c.push_back(121); 
// //      xy_c.push_back(424); 
// //      xy_c.push_back(424); 
// //      eclass['0'].push_back(xy_c);
// //      xy_c.push_back(42411);
// //      z.push_back(1123);
// //      eclass['1'].push_back(xy_c);
// //      eclass['0'].push_back(z);
     
// //     cout<<eclass['0'].size()<<endl;
// //     cout<<eclass['1'].size()<<endl;
// //     cout<<eclass['0'][0].size()<<endl;
// //     cout<<eclass['0'][1].size()<<endl;
// //     for(Eclass::iterator it=eclass.begin();it!=eclass.end();it++)
// //     {
// //        cout << "key:"<<it->first;
// //        for (vector<vector<int>>::iterator itv = (it->second).begin();itv != (it->second).end();itv++)//通过迭代器访问数据
// //         {
// // // 		//*it 代表的是vector<int>，又是一个容器，所以需要把小容器里边的数据遍历出来
// // // 		//小技巧：*it 就是vector 尖括号 里边的数据类型
// //             cout<<"num of vector"<<(it->second).size()<<endl;;
// //             for (vector<int>::iterator vit = (*itv).begin();vit != (*itv).end();vit++)
// //             {
// //                 cout<<":"<<*vit <<" ";
// //             }
// //         }
// //        cout<<endl;
            
    
// //     }
        
    



// //     return 0;
// // }

// 测试矩阵reshape
// #include<iostream>
// #include<vector>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <vector>
// #include<stdio.h>
// #include<stdlib.h>
// #include <fstream>
// using namespace std;

// vector<vector<int>> matrixReshape(vector<vector<int>>& nums, int r, int c) 
// {
//     int a=nums.size(),b=nums[0].size();
//     if(a*b!=r*c)
//     {
//        return nums;
//     }
//     vector<vector<int>> res(r,vector<int>(c));
//     for(int i=0;i<a*b;i++)
//     {
//        res[i/c][i%c]=nums[i/b][i%b];
//     }
//        return res;
// }
// int main()
// {
//     vector<vector<int>>s;
//     vector<int>a;
//     a.push_back(1);
//     a.push_back(2);
//     a.push_back(3);
//     a.push_back(4);
//     s.push_back(a);
//     cout<<s.size()<<endl;
//     matrixReshape(s,2,2);
//     cout<<s[0][0]<<" "<<s[0][1]<<endl;
//     return 0;
// }
// // int main()
// // {
// //     int a[10]={1,2,3,34,5,1,1,2,1,11};
// //     vector<vector<int>>p1(a,a+10);
// //     vector<string>s;
// //     s.push_back("你");
// //     s.push_back("好");
// //     s.push_back("帅");
// // //     vector<int>a,b,c;
// // //     a.push_back(1);
// // //     a.push_back(435);
// // //     a.push_back(3);
// // //     a.push_back(2);
// // //     b.push_back(111);
// // //     b.push_back(4351);
// // //     b.push_back(31);
// // //     b.push_back(21);
// // //     c.push_back(10);
// // //     c.push_back(4510);
// // //     c.push_back(310);
// // //     c.push_back(210);
// // //     p1.push_back(a);
// // //     p1.push_back(b);
// // //     p2.push_back(b);
// // //     p2.push_back(c);
    
    
// // //     cv::Mat coord_point=cv::Mat(p1,true); 
// // //     cout<<coord_point<<endl;
// // //     cout<<(int)(s.size())<<endl;
// // //     cout<<s[0]<<endl;
    
// // //     FILE *fp;
// // //     if((fp=fopen("/src/notebooks/ncnn-20220420/build/ceshichinese.txt","r"))==NULL)
// // //     {
// // //         printf("ERROR!\n");
// // //         exit(0);
// // //     }
// // //     char ch[3];   
// //     return 0;
// // }
// // //从txt文本读取中文汉字
// // std::vector<std::string> split_chinese(std::string s) {
// //     std::vector<std::string> t;
// //     for (size_t i = 0; i < s.length();)
// //     {
// //         int cplen = 1;
// //         if ((s[i] & 0xf8) == 0xf0)      // 11111000, 11110000
// //             cplen = 4;
// //         else if ((s[i] & 0xf0) == 0xe0) // 11100000
// //             cplen = 3;
// //         else if ((s[i] & 0xe0) == 0xc0) // 11000000
// //             cplen = 2;
// //         if ((i + cplen) > s.length())
// //             cplen = 1;
// //         t.push_back(s.substr(i, cplen));
// //         i += cplen;
// //     }
// //     return t;
// // }


// // int main()
// // {

    
// //     int acc=0,total=0;
// //     string s,image_path,label;
// //     ifstream inf;
// //     string result_rec;
// //     inf.open("/src/notebooks/ncnn-20220420/build/key_repvgg.txt");
// //     getline(inf,s);
// // //     while(getline(inf,s))
// // //     {  
// // //         int len_s=s.length();
// // //         cout<<len_s<<endl;
// // // //         cout<<s<<endl;
// // //         cout<<endl;
// // //         break;
// // //     }
// //     std::vector<std::string> t = split_chinese(s);
// //     cout<<t.size()<<endl;
// //     cout<<t[0]<<endl;
// //     return 0;
// // }



// ///////////////测试一下QVariant///////////
// // #include<iostream>
// // #include<QVariant>
// // using namespace std;

// // int main()
// // {
// //     QVariant var1="dahaidhalkd";
// //     cout<<var1.toString()<<endl;


// //     return 0;
// // }

#include<iostream>
#include<time.h>
// #include<stdio>
using namespace std;
int main()
{
    clock_t start,stop;   //定义两个clock函数返回类型
    start=clock();        //开始计时
    cout<<"hello world"<<endl;
    stop=clock();         //停止计时
    double time_response=((double)(stop-start))/CLOCKS_PER_SEC;    //这个值就是函数主体部分执行时间，单位为秒
    cout<<time_response<<endl;
    return 0; 
}

// #include "layer.h"
// #include "net.h"
// #include "simpleocv.h"
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/features2d/features2d.hpp>
// #include "opencv2/calib3d/calib3d.hpp" 
// #include <float.h>
// #include <stdio.h>
// #include <vector>
// #include <iostream>
// #include <algorithm>
// #include <string>
// #include <map>
// #include <list>
// #include <numeric>
// #include <fstream>
// #include <cmath>
// #include <exception>
// #include <time.h>
// #include <ctime>
// #include <regex>
// #include "hello.h"
// #include "operation.h"
// #include "yolov5.h"
// using namespace std;
// int main()
// {

//     say_hello();
//     int c1=1,c2=2;
//     cout<<add(c1,c2)<<endl;





//     return 0;
// }



