######通用版转换torch2onnx文件
######要求：保存的模型的方式是torch.save(model)即保存的是整个模型而不是state_dict()


#####0.加载配置文件

#####1.torch.load（“xxxxxxxxxxx”）

#####2.转换得到onnx模型
'''
    2.1.设置输入输出节点名
    2.2.给一个演练输入data，无关内容只关形状
    2.3.设置需要的动态维度
    2.4.torch.onnx.export
    2.5.是否需要simplify
'''

import yaml
import torch
import sys
import pdb
pdb.set_trace()


#0.
stream=open("./config_torch2onn.yaml",'r',encoding='utf-8')
config=yaml.load(stream,Loader=yaml.FullLoader)





#1.
file_path="../"+config["weight_path"].split("/")[1]
sys.path.append(file_path)
net=torch.load(config["weight_path"])
net.eval()

#2.1
input_name=['input']
output_name=['output']

#2.2
img=torch.zeros(config["input_img_size"])

#2.3
simplify=config["simplify"]
dynamic=config["dynamic"]
file_onnx=config["file_onnx"]


#2.4
torch.onnx.export(net,img,file_onnx,verbose=False,opset_version=11,export_params=True,
                  input_names=input_name,output_names=output_name,keep_initializers_as_inputs=True)


#2.5
import onnx
model_onnx=onnx.load(file_onnx)
onnx_check=onnx.checker.check_model(model_onnx)
if not onnx_check:
    print("onnx check is ok!")
    print("simplify:",simplify)
    if simplify: ######使用simplify简化onnx模型
        file=file_onnx.split(".onnx")[-2]+"_sim"+".onnx"
        print(file)
        try:
            import onnxsim
            print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                input_shapes={'images': list(img.shape)} if dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx,file)
        except Exception as e:
            print(f'simplifier failure: {e}')
#         print(f'{prefix} export success, saved as {file} ({file_size(file):.1file} MB)')
#         print(f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {file}'")
    else:
        print("no using simplify!!")       
else:
    print("onnx check is wrong!!!!")