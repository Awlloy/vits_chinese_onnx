import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer
import time
import argparse

parser = argparse.ArgumentParser(description='run test onnx model')
parser.add_argument('-b','--bert', type=str,default="./bert")
parser.add_argument('-o','--onnx', type=str,default="./onnx_model/prosody_model.onnx")
parser.add_argument('-t','--thread',type=int, default=11)
args = parser.parse_args()

model_dir = args.bert
o_pth = args.onnx

opt = ort.SessionOptions()
opt.intra_op_num_threads = args.thread  
opt.inter_op_num_threads = args.thread 


ort_session = ort.InferenceSession(o_pth,opt)
tokenizer = BertTokenizer.from_pretrained(model_dir)
dynamic_axes=False
text="你好，我是本次大赛的智能语音助手，小万，请问您有什么需要我帮忙的，你可以叫我的名字然后说出你的指令。"
input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


output_name = ["output"]

token_size = len(input_ids)
input_masks = [1] * token_size
type_ids = [0] * token_size
input_ids = np.expand_dims(np.array(input_ids),0)
input_masks = np.expand_dims(np.array(input_masks),0)
type_ids = np.expand_dims(np.array(type_ids),0)
input_data = [input_ids,input_masks,type_ids]

input_name = ["input_ids","input_masks","type_ids"]
input_ = dict(zip(input_name,input_data))
t = time.time()

result = ort_session.run(None, input_)
print("run %.2fms"%((time.time()-t) * 1000))
print(result)
