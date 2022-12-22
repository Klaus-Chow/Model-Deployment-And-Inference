####1.加载训练得到的权重
# import pdb
# pdb.set_trace()
import torch
device=torch.device('cpu')
weight_path="../../../BankRecogProj_20220506/dbnet/mobile.pth.tar"
checkpoint=torch.load(weight_path,device) ##加载权重



####2.加载网络
import yaml
import sys
sys.path.append("../../../BankRecogProj_20220506/dbnet")
from models.DBNet import DBNet
stream = open("../../../BankRecogProj_20220506/dbnet/config_mobile.yaml",'r',encoding="utf-8")
config=yaml.load(stream,Loader=yaml.FullLoader)
config['train']['pretrained']=config['test']['pretrained']
db_model=DBNet(config)




####3.网络加载模型参数
model_dict=checkpoint
db_model.load_state_dict(model_dict)
db_model.eval()



####4.转换得到onnx模型
'''
    4.1.设置输入输出节点名
    4.2.给出一个演练输入data(无关内容，形状确定即可，常常使用torch.zeros全零矩阵来演练)
    4.3.设置需要的动态维度
    4.4.torch.onnx.export转换
    4.5 是否需要进行simplify 
'''
#4.1
input_name=['input']
output_name=['output']

#4.2
img=torch.zeros(1,3,320,320)

#4.3
dynamic_axes={"input":[2,3],'output':[2,3]}######若不需要动态输入改成false
dynamic=True
simplify = not dynamic
file_onnx="../onnx_models/dbnet/dbnet_mobile_dynamic.onnx"

#4.4    
# torch.onnx.export(db_model,img,file_onnx,verbose=False,opset_version=11,
#                  input_names=input_name,output_names=output_name,
#                  dynamic_axes=dynamic_axes)

torch.onnx.export(db_model,img,file_onnx,verbose=False,opset_version=11,
                 input_names=input_name,output_names=output_name,dynamic_axes=dynamic_axes)


#4.5
import onnx
model_onnx=onnx.load(file_onnx)
onnx_check=onnx.checker.check_model(model_onnx)
if not onnx_check:
    print("onnx check is OK!!!")
    print("simplify:",simplify)
    if simplify:
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
    print("onnx check is wrong!!!!!")










