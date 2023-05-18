from vits_pinyin import VITS_PinYin
import utils
from text.symbols import symbols
import torch
import onnx
from text import cleaned_text_to_sequence
import onnxsim
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='export onnx model for stream infer')
parser.add_argument('-m','--model', type=str,default="./model/vits_bert_model.pth")
parser.add_argument('-b','--bert', type=str,default="./bert")
parser.add_argument('-c','--config', type=str,default="./configs/bert_vits.json")
parser.add_argument('-o','--output', type=str,default="./onnx_model/vits_bert_encode.onnx")
parser.add_argument('-s','--simplify',action='store_true', default=False)
parser.add_argument('-p','--opset',type=int, default=11)


args = parser.parse_args()
simplify = args.simplify
bert = args.bert
config = args.config
i_pth = args.model
o_pth = args.output
device="cpu"
opset = args.opset
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
class ExportModel(torch.nn.Module):
    
    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model=model
    def forward(self, x, x_lengths, bert, noise_scale=1.0, length_scale=1.0,sid=None,max_len=None):
        return self.model.infer(x, x_lengths, bert, sid=sid, noise_scale=noise_scale, length_scale=length_scale, max_len=max_len)[0]
class ExportPauseModel(torch.nn.Module):
    
    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model=model
    def forward(self, x, x_lengths, bert,pause_mask,pause_value, noise_scale=1.0, length_scale=1.0,sid=None,max_len=None):
        return self.infer_pause(x, x_lengths, bert,pause_mask=pause_mask,pause_value=pause_value, sid=sid, noise_scale=noise_scale, length_scale=length_scale, max_len=max_len)[0]
    
# input("run")
mode = "infer"
with torch.no_grad():
    x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
    x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
    # audio = net_g(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5,
    #                     length_scale=1)[0][0, 0].data.cpu().float().numpy()
    noise_scale=torch.FloatTensor([0.5]).to(device)
    length_scale=torch.FloatTensor([1.0]).to(device)
    
    dynamic_axes = {
        'x_tst': {1: 'T_S'},
        'x_tst_prosody': {1: 'T_S'}
    }
    if mode=="infer_pause":# infer_pause
        model = ExportPauseModel(net_g)
        pause_tmpt = np.array(input_ids)
        print(len(pause_tmpt),input_ids)
        pause_mask = np.where(pause_tmpt == 2, 0, 1)
        pause_value = np.where(pause_tmpt == 2, 1, 0)
        input_name = ["x_tst","x_tst_lengths","x_tst_prosody","pause_mask", "pause_value","noise_scale","length_scale"]
        output = model(x_tst, x_tst_lengths, x_tst_prosody,pause_mask=pause_mask,pause_value=pause_value, noise_scale=noise_scale,
                        length_scale=length_scale)
    else:
        model = ExportModel(net_g)
        input_name = ["x_tst","x_tst_lengths","x_tst_prosody","noise_scale","length_scale"]
        output = model(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=noise_scale,
                        length_scale=length_scale)
        
    output_name = ["output"]
    if isinstance(output,np.ndarray) and len(output.shape)>1:
        output=output[0, 0].data.cpu().float().numpy()
    else:
    # print(output,len(output),output.shape,type(output))
        audio=output
    print("start export")
    torch.onnx.export(
        model,(x_tst, x_tst_lengths, x_tst_prosody,noise_scale,length_scale), o_pth, 
        opset_version=opset,
        export_params=True, 
        do_constant_folding=True,
        input_names=input_name,output_names=output_name,
        dynamic_axes=dynamic_axes, 
        verbose=False)
    print("export done")
    print("test load model")

    onnx_model = onnx.load(o_pth)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    if simplify:
        print("export simplify")
        simplify_onnx_model, check = onnxsim.simplify(onnx_model)
        # onnx.save(onnx.shape_inference.infer_shapes(simplify_onnx_model), "./onnx_model_simplify_temp.onnx")
        onnx.save(simplify_onnx_model, o_pth)

print("save done")
    
