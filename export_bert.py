from bert.ProsodyModel import TTSProsody
import torch
import onnx
import onnxsim
import argparse

parser = argparse.ArgumentParser(description='export bert onnx model')
parser.add_argument('-b','--bert', type=str,default="./bert")
parser.add_argument('-o','--output', type=str,default="./onnx_model/prosody_model.onnx")
parser.add_argument('-s','--simplify',action='store_true',default=False)
parser.add_argument('-p','--opset',type=int, default=11)
args = parser.parse_args()


pth = args.bert
o_pth = args.output
device='cpu'

ttsmodel = TTSProsody(pth,device)
simplify=args.simplify
dynamic_axes=False
opset = args.opset

text="你好，请问您有什么需要我帮忙的。"
input_ids = ttsmodel.char_model.text2Token(text)

token_size = len(input_ids)
input_masks = [1] * token_size
type_ids = [0] * token_size
# print(l,input_ids)
print(input_ids,input_masks)
with torch.no_grad():
    input_ids = torch.LongTensor([input_ids]).to(device)
    input_masks = torch.LongTensor([input_masks]).to(device)
    type_ids = torch.LongTensor([type_ids]).to(device)
    # summary(ttsmodel.char_model, *(input_ids, input_masks, type_ids))
    output = ttsmodel.char_model(input_ids, input_masks, type_ids)
    print(output)
    # inputs_masks
    # tokens_type_ids
    input_name = ["input_ids","input_masks","type_ids"]
    output_name = ["output"]
    dynamic_axes = {
            'input_ids': {1: 'T_S'},
            'input_masks': {1: 'T_S'},
            'type_ids': {1: 'T_S'},
            'output': {1: 'T_S'}
        }
    print("start export")
    torch.onnx.export(
        ttsmodel.char_model,(input_ids, input_masks, type_ids), o_pth, opset_version=opset,
        export_params=True, do_constant_folding=True,
        input_names=input_name,output_names=output_name,
        dynamic_axes=dynamic_axes, 
        verbose=False)
    print("export done")
    print("test load model")

    onnx_encoder = onnx.load(o_pth)
    onnx.checker.check_model(onnx_encoder)
    onnx.helper.printable_graph(onnx_encoder.graph)
    if simplify:
        print("export simplify")
        simplify_onnx_model, check = onnxsim.simplify(onnx_encoder)
        onnx.save(simplify_onnx_model, o_pth)
print("save done")
