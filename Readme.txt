///////////////////----------------------------------------项目架构


build--------------------编译后的可执行文件



                       |-----Crnn.h: 提供一个文字识别器类，其构造函数用于初始化ncnn::Net crnn，加载model.param和model.bin, 还有一个成员函数                                   detect_crnn用于推理。
                  
         |--------Crnn-------|-----Utils_crnn.h：提供crnn推理中需要用到的工具函数
        
                    
         |-------DBNet-------|-----DBnet.h: 提供了一个文字检测器类，其构造函数用于初始化ncnn::Net dbnet,加载model.param和model.bin,还有一个成员函                                   数detect_text用于推理。
                       |-----Utils_dbnet.h：提供了dbnet推理中需要用到的后处理函数
                  
                  
Include-----|----ModelService----|-----ModelService.h: 提供一个modelservice构造函数,通过类的组合，添加Crnn、DBnet、Yolov5的类定义的对象作为成员变量                                        （这样的好处就是需要用到其他算法时只要添加相应的类的对象作为成员即可）

         |-----Utils----------|-----Tools.h：里面提供的是串联整个项目所需要的工具函数，比如抠图、透视变换、画图、识别后对文本修正的后处理等。
        
        
         |-----Yolov5---------|-----Utils_yolo.h：提供yolov5推理中需要用到的工具函数，比如生成候选框、排序、nms、IOU计算等。
                       
                        |-----Yolov5.h: 提供一个目标检测器类，其构造函数用于初始化ncnn::Net yolov5，加载model.param和model.bin, 还有一个成                                    员函数detect用于推理。



Src：Include中每一个.h所对应的.cpp源文件

Output:保存中间结果

CMakeLists.txt：编译文件，自定义了函数ncnn_add_example，加载opencv、加载ncnn、加载所有源文件src、生成可执行文件并link到opencv和ncnn、include相关的头文件目录

main_crnn.cpp   用于测试crnn推理

main_dbnet.cpp   用于测试dbnet推理

main_yolo.cpp   用于测试yolo推理,yolov5中的focus自定义添加到ncnn中编译。

main_modelservice.cpp   用于身份证项目



/************************************************2022.8.14  更新torch转onnx ****************/
涉及算法yolov5、dbnet、crnn
2022/8/16
torch_onnx ：添加了yolo_export: export和test_yolov5s_onnx.ipynb适用于所有yolo系列
onnx_models：yolov5存放相关的onnx模型
onnx_test_output: 存放onnx模型推理的结果

Remind:注意使用onnxruntime推理onnx模型时，动态输入输出和静态输入输出的区别


/************************************************2022.8.17 更新onnx转ncnn ******************/

Remind:注意使用ncnn部署模型时，ncnn支持动态输入输出，所以无论onnx是静态输入还是动态输入，在转成ncnn框架后都是允许动态输入的，但是在param中需要将输出改成动态‘-1’，所以建议使用ncnn部署时，使用onnx静态，这样可以使用onnxsim进行结构优化，当然还有一些特殊的op得自己改

yolov5中的focus层在onnx中无法直接转换成ncnn，所以在src/layer中自定义了yolov5focus.h和yolov5focus.cpp
使用./onnx2ncnn将onnx模型转换成ncnn,cd ncnn-20220420/build/tools/onnx ./onnx2ncnn best-lite.onnx best-lite.param best-lite.bin
使用./ncnnoptimize优化网络结构，加速模型的推理

ncnn fp32 加速优化 cd ncnn-20220420/build/tools ./ncnnoptimize best-lite.param best-lite.bin best-lite_fp32.param best-lite_fp32.bin 0



ncnn fp16量化 cd ncnn-20220420/build/tools ./ncnnoptimize best-lite.param best-lite.bin best-lite_fp16.param best-lite_fp16.bin 65536





ncnn int8量化 cd ncnn-20220420/build/tools ./ncnnoptimize best-lite.param best-lite.bin best-lite_fp32.param best-lite_fp32.bin 0
          准备相关校验（论文说1000+左右数据，从训练集中抽取）
          find images/ -type f > imagelist.txt  获取相关list
          cd ncnn-20220420/build/tools/quantize ./ncnn2table best-lite_fp32.param best-lite_fp32.bin /src/notebooks/IDimg/imagelist.txt best-lite-mtz.table mean=[0,0,0] norm=[1/255.0,0.0039215,0.0039215] shape=[640,640,3],pixel=BGR thread=8 method=aciq
          再使用 ./ncnn2int8 best-lite_fp32.param best-lite_fp32.bin best-lite_int8.param best-lite_int8.bin best-lite-mtz.table







/*************************************************2022.8.15 更新剪枝************************/


