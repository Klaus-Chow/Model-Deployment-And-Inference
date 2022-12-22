import pdb
pdb.set_trace()
######输入[1，3,32,280]
import sys
import torch
#s1加载参数权重
sys.path.append("../../../crnnmobile")
device=torch.device("cpu")
weight_path="../../../crnnmobile/model/mobile_fix.pth"
checkpoint=torch.load(weight_path,device) ###加载参数权重
checkpoint.keys()



#s2加载模型
from keys_sythtext import alphabet
from models.repvgg import repvgg_model_convert
from models.crnn_mobilev3_repvgg import CRNNSmallNoLSTM,CRNNLargeNoLSTM,CRNNSmallLSTM,CRNNLargeLSTM


nclass=len(alphabet)+1
print(nclass)

net=CRNNSmallNoLSTM(128,nclass,1)

net = repvgg_model_convert(net,None)



#3.模型加载权重
net.load_state_dict(checkpoint)
# model_dict = {}
# for key,value in checkpoint.items():
#     model_dict[key[7:]]=value
net.eval()



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

# 4.2
img=torch.zeros(1,3,32,280)

#4.3
simplify=True
dynamic=False
file_onnx="../onnx_models/crnn/id_crnn_mobile_fix.onnx"  ####输出的onnx模型

#4.4
# dynamic_axes={'input':[3],'output':[2,3]}####动态尺寸
torch.onnx.export(net,img,file_onnx,verbose=False,opset_version=11,export_params=True,
                  input_names=input_name,output_names=output_name,keep_initializers_as_inputs=True)

#4.5
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