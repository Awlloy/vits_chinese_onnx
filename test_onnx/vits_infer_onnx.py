import os
import numpy as np


from scipy.io import wavfile
from text.symbols import symbols
from text import cleaned_text_to_sequence
from text import pinyin_dict
import onnxruntime as ort
import pypinyin
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin
from transformers import BertTokenizer
import time
import argparse
class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
class VITS_PinYin:
    def __init__(self, bert_path,onnx_model):
        self.pinyin_parser = Pinyin(MyConverter())
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1
        opt.inter_op_num_threads = 1
        self.prosody = ort.InferenceSession(onnx_model,opt)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
    def text2Token(self, text):
        token = self.tokenizer.tokenize(text)
        txtid = self.tokenizer.convert_tokens_to_ids(token)
        return txtid
    def get_char_embeds(self, text):
        input_ids = self.text2Token(text)
        
        
        token_size = len(input_ids)
        input_masks = [1] * token_size
        type_ids = [0] * token_size
        input_ids = np.expand_dims(np.array(input_ids),0)
        input_masks = np.expand_dims(np.array(input_masks),0)
        type_ids = np.expand_dims(np.array(type_ids),0)
        char_embeds = self.prosody.run(None,{"input_ids":input_ids,"input_masks":input_masks,"type_ids":type_ids})
        return char_embeds

    def expand_for_phone(self, char_embeds, length):  # length of phones for char
        assert char_embeds.shape[0] == len(length)
        expand_vecs = list()
        for vec, leng in zip(char_embeds, length):
            vec = np.broadcast_to(vec, (leng, *vec.shape))
            expand_vecs.append(vec)
        expand_embeds = np.concatenate(expand_vecs, axis=0)
        assert expand_embeds.shape[0] == sum(length)
        return expand_embeds
    
    def chinese_to_phonemes(self, text):
        # @todo:考虑使用g2pw的chinese bert替换原始的pypinyin,目前测试下来运行速度太慢。
        # 将标准中文文本符号替换成 bert 符号库中的单符号,以保证bert的效果.
        text = text.replace("——", "...")\
            .replace("—", "...")\
            .replace("……", "...")\
            .replace("…", "...")\
            .replace('“', '"')\
            .replace('”', '"')\
            .replace("\n", "")
        tokens = self.tokenizer.tokenize(text)
        text = ''.join(tokens)
        assert not tokens.count("[UNK]")
        pinyins = np.reshape(pypinyin.pinyin(text, style=pypinyin.TONE3), (-1))
        try:
            phone_index = 0
            phone_items = []
            phone_items.append('sil')
            count_phone = []
            count_phone.append(1)
            temp = ""

            len_pys = len(tokens)
            for word in tokens:
                if is_chinese(word):
                    count_phone.append(2)
                    if (phone_index >= len_pys):
                        print(
                            f"!!!![{text}]plz check ur text whether includes MULTIBYTE symbol.\
                                (请检查你的文本中是否包含多字节符号)")
                    pinyin = pinyins[phone_index]
                    phone_index = phone_index + 1
                    if not pinyin[-1].isdigit():
                        pinyin += "5"
                    if pinyin[:-1] in pinyin_dict:
                        tone = pinyin[-1]
                        a = pinyin[:-1]
                        a1, a2 = pinyin_dict[a]
                        phone_items += [a1, a2 + tone]
                else:
                    temp += word
                    if temp == pinyins[phone_index]:
                        temp = ""
                        phone_index += 1
                    count_phone.append(1)
                    phone_items.append('sp')

            count_phone.append(1)
            phone_items.append('sil')
            phone_items_str = ' '.join(phone_items)
        except IndexError as e:
            print('except:', e)

        text = f'[PAD]{text}[PAD]'
        char_embeds = self.get_char_embeds(text)[0][0]
        char_embeds = self.expand_for_phone(char_embeds, count_phone)
        return phone_items_str, char_embeds


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))

parser = argparse.ArgumentParser(description='run test onnx model')
parser.add_argument('-b','--bert', type=str,default="./bert")
parser.add_argument('--bert_onnx', type=str,default="./onnx_model/prosody_model.onnx")
parser.add_argument('--vits_onnx', type=str,default="./onnx_model/vits_bert_model.onnx")
parser.add_argument('-t','--thread',type=int, default=11)
args = parser.parse_args()

bert = args.bert
bert_onnx = args.bert_onnx
vits_bert_onnx = args.vits_onnx

opt = ort.SessionOptions()
opt.intra_op_num_threads = args.thread  
opt.inter_op_num_threads = args.thread 
# pinyin
tts_front = VITS_PinYin(bert,bert_onnx)

net_g = ort.InferenceSession(vits_bert_onnx,opt)


os.makedirs("../vits_infer_out/", exist_ok=True)
if __name__ == "__main__":
    n = 0
    fo = open("../vits_infer_item.txt", "r+", encoding='utf-8')
    while (True):
        try:
            item = fo.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (item == None or item == ""):
            break
        print("start read ")
        t = time.time()
        n = n + 1
        phonemes, char_embeds = tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)
        print(len(input_ids))
        x_tst = np.expand_dims(np.array(input_ids),0)
        x_tst_lengths = np.array([len(input_ids)])
        x_tst_prosody = np.expand_dims(np.array(char_embeds),0)
        noise_scale=np.array([0.5],np.float32)
        length_scale=np.array([1.0],np.float32)
        output = net_g.run(None,{
            "x_tst":x_tst,"x_tst_lengths":x_tst_lengths,"x_tst_prosody":x_tst_prosody,"noise_scale":noise_scale,"length_scale":length_scale})
        output=output[0]
        audio = output[0,0].astype(np.float32)
        print("run time ",time.time()-t,' s')
        save_wav(audio, f"../vits_infer_out/bert_vits_{n}.wav", 16000)
    fo.close()
