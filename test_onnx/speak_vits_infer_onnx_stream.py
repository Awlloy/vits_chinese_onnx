import os
import numpy as np


from scipy.io import wavfile
from text.symbols import symbols
from text import cleaned_text_to_sequence
from text import pinyin_dict
import onnxruntime as ort
import pyaudio

import pypinyin
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin
from transformers import BertTokenizer
import time

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



# pinyin
tts_front = VITS_PinYin("../bert","../onnx_model/prosody_model.onnx_simplify.onnx")

# config
encode_pth = "../onnx_model/vits_bert_encode.onnx_simplify.onnx"
decode_pth = "../onnx_model/vits_bert_decode.onnx_simplify.onnx"
opt = ort.SessionOptions()
opt.intra_op_num_threads = 1  # 设置为 1，强制使用单核推理
opt.inter_op_num_threads = 1  # 设置为 1，强制
encode_model = ort.InferenceSession(encode_pth,opt)
decode_model = ort.InferenceSession(decode_pth,opt)



def decode_stream(z,decode_model):
    print("-----------------------------------------------------------------------------")
    import datetime
    import numpy
    print(datetime.datetime.now())
    # z, attn, y_mask, (z, z_p, m_p, logs_p),g = self.encode(x, x_lengths, bert, sid=sid, noise_scale=noise_scale, length_scale=length_scale)
    t = time.time()
    len_z = z.shape[2]
    print('frame size is: ', len_z)
    if (len_z < 100):
        print('no nead steam')
        # one_time_wav = self.dec(z, g=g)[0, 0].data.cpu().float().numpy()
        # one_time_wav = self.decode(z,g=g)[0, 0].data.cpu().float().numpy()
        one_time_wav = decode_model.run(None,{"input":z})[0][0,0]
        print("first speak ",time.time()-t)
        return one_time_wav

    # can not change these parameters
    hop_length = 256 # bert_vits.json
    hop_frame = 9
    hop_sample = hop_frame * hop_length
    
    stream_chunk = 30
    stream_index = 0
    stream_out_wav = []

    while (stream_index + stream_chunk < len_z):
        if (stream_index == 0): # start frame
            cut_s = stream_index
            cut_s_wav = 0
        else:
            cut_s = stream_index - hop_frame
            cut_s_wav = hop_sample

        if (stream_index + stream_chunk > len_z - hop_frame): # end frame
            cut_e = stream_index + stream_chunk
            cut_e_wav = 0
        else:
            cut_e = stream_index + stream_chunk + hop_frame
            cut_e_wav = -1 * hop_sample
        
        z_chunk = z[:, :, cut_s:cut_e]
        # o_chunk = self.dec(z_chunk, g=g)[0, 0].data.cpu().float().numpy()
        # o_chunk = self.decode(z_chunk,g=g)[0, 0].data.cpu().float().numpy()
        o_chunk = decode_model.run(None,{"input":z_chunk})[0][0,0]
        
        o_chunk = o_chunk[cut_s_wav:cut_e_wav]
        if stream_index==0:
            print("first speak ",time.time()-t)
        stream_out_wav.extend(o_chunk)
        stream_index = stream_index + stream_chunk
        # print(datetime.datetime.now())

    if (stream_index < len_z):
        cut_s = stream_index - hop_frame
        cut_s_wav = hop_sample
        z_chunk = z[:, :, cut_s:]
        # o_chunk = self.dec(z_chunk, g=g)[0, 0].data.cpu().float().numpy()
        # o_chunk = self.decode(z_chunk,g=g)[0, 0].data.cpu().float().numpy()
        o_chunk = decode_model.run(None,{"input":z_chunk})[0][0,0]
        o_chunk = o_chunk[cut_s_wav:]
        stream_out_wav.extend(o_chunk)
        if stream_index==0:
            print("first speak ",time.time()-t)

    stream_out_wav = numpy.asarray(stream_out_wav)
    return stream_out_wav
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
            print("item is None")
            break
        print("start read ")
        t = time.time()
        n = n + 1
        phonemes, char_embeds = tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)
        x_tst = np.expand_dims(np.array(input_ids),0)
        x_tst_lengths = np.array([len(input_ids)])
        x_tst_prosody = np.expand_dims(np.array(char_embeds),0)
        noise_scale=np.array([0.5],np.float32)
        length_scale=np.array([1.0],np.float32)
        encode_output = encode_model.run(None,{
            "x_tst":x_tst,"x_tst_lengths":x_tst_lengths,"x_tst_prosody":x_tst_prosody,"noise_scale":noise_scale,"length_scale":length_scale})
        z,mask=encode_output
        print('start speek ',time.time()-t)
        audio=decode_stream(z,decode_model)
        # output=
        # audio = output[0,0].astype(np.float32)
        print("run time ",time.time()-t)
        save_wav(audio, f"../vits_infer_out/bert_vits_stream{n}.wav", 16000)
    fo.close()
