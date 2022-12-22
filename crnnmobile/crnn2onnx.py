######输入[1，3,32,280]
import sys
import torch
sys.path.append("../")
device=torch.device("cpu")
weight_path="crnnmobile/model/mobile_fix.pth"
checkpoint=torch.load(weight_path,device) ###加载参数权重
checkpoint.keys()



#加载模型
from keys_sythtext import alphabet
from models.repvgg import repvgg_model_convert
from models.crnn_mobilev3_repvgg import CRNNSmallNoLSTM,CRNNLargeNoLSTM,CRNNSmallLSTM,CRNNLargeLSTM


nclass=len(alphabet)+1
print(nclass)

net=CRNNSmallNoLSTM(128,nclass,1)

net = repvgg_model_convert(net,None)
net.load_state_dict(checkpoint)

# model_dict = {}
# for key,value in checkpoint.items():
#     model_dict[key[7:]]=value
net.eval()




####获取onnx模型
input_name=['input']
output_name=['output']
img=torch.zeros(1,3,32,280)
# dynamic_axes={'input':[3],'output':[2,3]}####动态尺寸
torch.onnx.export(net,img,"./crnnmobile/id_crnn_mobile_fix.onnx",verbose=False,opset_version=11,export_params=True,
                  input_names=input_name,output_names=output_name,keep_initializers_as_inputs=True)