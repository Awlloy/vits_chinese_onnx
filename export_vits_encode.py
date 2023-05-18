
from vits_pinyin import VITS_PinYin
import utils
from text.symbols import symbols
import torch
import onnx
from bert.ProsodyModel import TTSProsody
from text import cleaned_text_to_sequence
import onnxsim
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='export onnx model for stream infer')
parser.add_argument('-m','--model', type=str,default="./model/vits_bert_model.pth")
parser.add_argument('-b','--bert', type=str,default="./bert")
parser.add_argument('-c','--config', type=str,default="./configs/bert_vits.json")
parser.add_argument('-o','--output', type=str,default="./onnx_model/vits_bert_encode.onnx")
parser.add_argument('-d','--output_decode',type=str,default="./onnx_model/vits_bert_decode.onnx")
parser.add_argument('-s','--simplify',action='store_true', default=False)
parser.add_argument('-p','--opset',type=int, default=11)
args = parser.parse_args()

simplify = args.simplify

bert = args.bert

config = args.config
i_pth = args.model
o_encode_pth = args.output
o_decode_pth = args.output_decode
opset = args.opset

device="cpu"

hps = utils.get_hparams_from_file(config)

tts_front = VITS_PinYin(bert, device)
text="你好，请问您有什么需要我帮忙的。"
# model
net_g = utils.load_class(hps.train.eval_class)(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
utils.load_model(i_pth, net_g)
net_g.eval()
net_g.to(device)

phonemes, char_embeds = tts_front.chinese_to_phonemes(text)
input_ids = cleaned_text_to_sequence(phonemes)
# input("run")
class ExportEncodeModel(torch.nn.Module):

    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model=model
    def forward(self, x, x_lengths, bert, noise_scale=1.0, length_scale=1.0,sid=None):
        return self.model.encode(x, x_lengths, bert, sid=sid, noise_scale=noise_scale, length_scale=length_scale)
class ExportDecodeModel(torch.nn.Module):
    
    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model=model
    def forward(self, x,g=None):
        return self.model.decode(x,g)

def export_onnx(model,input_data,output_path,export_params=True,do_constant_folding=True,opset_version=11,dynamic_axes=None,input_names=None,output_names=None,verbose=False):
    print("start export")
    torch.onnx.export(
        model,input_data, output_path, 
        opset_version=opset_version,
        export_params=export_params, 
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes, 
        verbose=verbose)
    print("export done")
    print("test load model")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    print("test load model done")
    
    
encode_model = ExportEncodeModel(net_g)
decode_model = ExportDecodeModel(net_g)

with torch.no_grad():
    x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
    x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
    noise_scale=torch.FloatTensor([0.5]).to(device)
    length_scale=torch.FloatTensor([1.0]).to(device)
    
    dynamic_axes = {
        'x_tst': {1: 'T_S'},
        'x_tst_prosody': {1: 'T_S'}
    }
    input_name = ["x_tst","x_tst_lengths","x_tst_prosody","noise_scale","length_scale"]
    encode_output = encode_model(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=noise_scale,
                    length_scale=length_scale)
    z,mask,g=encode_output
    output_name = ["output","mask"]
    print("start export encode")
    export_onnx(
        encode_model,(x_tst, x_tst_lengths, x_tst_prosody,noise_scale,length_scale), o_encode_pth, 
        opset_version=opset,
        export_params=True, 
        do_constant_folding=True,
        input_names=input_name,
        output_names=output_name,
        dynamic_axes=dynamic_axes, 
        verbose=False)
    print("export  encode done ")
    print("test load encode model ")

    onnx_model = onnx.load(o_encode_pth)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    if simplify:
        print("export simplify")
        simplify_onnx_model, check = onnxsim.simplify(onnx_model)
        # onnx.save(onnx.shape_inference.infer_shapes(simplify_onnx_model), "./onnx_model_simplify_temp.onnx")
        onnx.save(simplify_onnx_model, o_encode_pth)
    if "onnx" in o_decode_pth:
        print("try export decode")
        dynamic_axes = {
            'input': {2: 'T_S'}
        }
        input_name = ["input"]
        output_name= ["output"]
        z_t = z.to(device)
        print("start export decode")
        export_onnx(
            decode_model,(z_t), o_decode_pth, 
            opset_version=opset,
            export_params=True, 
            do_constant_folding=True,
            input_names=input_name,
            output_names=output_name,
            dynamic_axes=dynamic_axes, 
            verbose=False)
        print("export decode  done")
        print("test load  decode model")

        onnx_model = onnx.load(o_decode_pth)
        onnx.checker.check_model(onnx_model)
        onnx.helper.printable_graph(onnx_model.graph)
        if simplify:
            print("export simplify")
            simplify_onnx_model, check = onnxsim.simplify(onnx_model)
            onnx.save(simplify_onnx_model, o_decode_pth)
print("export done")
    
