import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(parent_path)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
print(tokenizer)
# tokenizer.padding_side = 'right'
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)
texts = ['尽管中华民国临时政府已于南京成立，然而位于北京的大清政府仍继续存在，并以时任内阁总理大臣袁世凯及其北洋军为主力持续与革命党人对抗','法国的首都在巴黎。']
input_ids= tokenizer(texts,padding=True,add_special_tokens=False)
print(input_ids)
print(tokenizer.bos_token_id)
print(tokenizer.eos_token)
print(tokenizer.bos_token)
print(tokenizer('尽管中华民国'))
print(tokenizer('尽管'))
print(tokenizer('中华民国'))
print(tokenizer.encode('尽管中华民国',add_special_tokens=False))

from langdetect import detect
from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.CHINESE]
detector = LanguageDetectorBuilder.from_languages(*languages).build()
text ='GLM-4-9B 是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源版本。 在语义、数学、推理、代码和知识等多方面的数据集测评中，GLM-4-9B 及其人类偏好对齐的版本 GLM-4-9B-Chat 均表现出较高的性能。 除了能进行多轮对话，GLM-4-9B-Chat 还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大 128K 上下文）等高级功能。 本代模型增加了多语言支持，支持包括日语，韩语，德语在内的 26 种语言。我们还推出了支持 1M 上下文长度（约 200 万中文字符）的模型。'
import harvesttext as ht
r = detect(text)
language = detector.detect_language_of(text)
print(language)
print(r)
ht = ht.HarvestText()
sentences = ht.cut_sentences(text)
print(sentences)

text = 'GLM-4-9B is the open-source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI. In the evaluation of data sets in semantics, mathematics, reasoning, code, and knowledge, GLM-4-9B and its human preference-aligned version GLM-4-9B-Chat have shown superior performance beyond Llama-3-8B. In addition to multi-round conversations, GLM-4-9B-Chat also has advanced features such as web browsing, code execution, custom tool calls (Function Call), and long text reasoning (supporting up to 128K context). This generation of models has added multi-language support, supporting 26 languages including Japanese, Korean, and German. We have also launched the GLM-4-9B-Chat-1M model that supports 1M context length (about 2 million Chinese characters) and the multimodal model GLM-4V-9B based on GLM-4-9B. GLM-4V-9B possesses dialogue capabilities in both Chinese and English at a high resolution of 1120*1120. In various multimodal evaluations, including comprehensive abilities in Chinese and English, perception & reasoning, text recognition, and chart understanding, GLM-4V-9B demonstrates superior performance compared to GPT-4-turbo-2024-04-09, Gemini 1.0 Pro, Qwen-VL-Max, and Claude 3 Opus.'
import nltk
sentences = nltk.sent_tokenize(text)
print(sentences)
r = detect(text)
language = detector.detect_language_of(text)
print(language)
print(r)
print(tokenizer.additional_special_tokens)
print(tokenizer.additional_special_tokens_ids)
print(tokenizer.all_special_tokens_extended)
print(tokenizer.decode([2082, 1113, 2142, 374, 264, 4948, 19092, 323, 7203, 429, 374, 65868, 938, 315, 11193, 323, 59003, 678, 89756, 11, 53901, 533, 7586, 315, 28819, 13, 2082, 1113, 2142, 6738, 369, 279, 74750, 315, 279, 1584, 11, 892, 432, 9977, 311, 387, 25088, 11, 75637, 11, 323, 27662, 13, 2121, 264, 34628, 2115, 28281, 7203, 11, 9096, 389, 279, 3041, 60417, 2115, 315, 279, 4948, 19708, 11, 432, 374, 5990, 7481, 16249, 56455, 2142, 323, 55647, 82031, 438, 279, 55647, 19974, 320, 2740, 529, 8820, 50186, 8, 315, 279, 39968, 7203, 11, 323, 702, 264, 3746, 13647, 14998, 448, 7147, 96317, 2142, 323, 50186, 13, 93771, 12157, 304, 33513, 2041, 15896, 12406, 1113, 550, 1293, 1573, 279, 21221, 315, 15896, 5302, 11, 75855, 11, 476, 976, 18933, 13, 2354, 279, 9995, 315, 38127, 69373, 12859, 11, 65868, 41578, 8840, 11193, 1083, 15996, 13, 15790, 34213, 315, 76713, 3381, 525, 1730, 6814, 3840, 11, 6481, 43216, 2142, 21938, 504, 279, 91060, 13, 16001, 279, 15259, 4279, 315, 279, 220, 98729, 339, 323, 279, 1156, 10788, 315, 279, 220, 98360, 339, 9291, 11, 279, 76713, 7203, 19790, 3304, 304, 1429, 5479, 315, 279, 1879, 323, 1030, 264, 5089, 3476, 304, 7337, 6, 27775, 369, 89381, 48282, 13, 72083, 76713, 8681, 315, 3381, 14113, 2337, 419, 4168, 13, 2082, 1113, 1671, 614, 4429, 949, 304, 3807, 91697, 11, 1429, 33881, 304, 279, 12089, 6804, 2886, 11, 279, 8521, 16384, 5004, 323, 279, 15142, 16384, 5004, 11, 6693, 835, 12857, 279, 835, 315, 279, 28721, 11380, 315, 43216, 2142, 13, 641, 279, 1537, 10788, 315, 279, 220, 98360, 339, 323, 1119, 279, 220, 99146, 267, 9291, 11, 279, 76713, 7203, 702, 1012, 592, 84355, 3055, 803, 13, 2082, 1113, 2142, 49820, 264, 19455, 315, 25264, 304, 1973, 311, 3367, 1181, 10502, 10330, 892, 646, 387, 42684, 18630, 1119, 28984, 323, 40666, 25264, 26, 1052, 374, 5089, 27159, 1948, 279, 1378, 11, 892, 525, 16220, 52495, 13, 36001, 3214, 658, 25264, 9210, 311, 4446, 1495, 11193, 323, 1584, 11, 3432, 4429, 264, 16387, 2484, 304, 279, 3267, 11, 1393, 40666, 25264, 9210, 311, 855, 17755, 1128, 458, 76713, 8231, 1035, 387, 1075, 13, 2082, 1113, 380, 3381, 11, 18784, 11, 323, 548, 7184, 614, 6342, 264, 949, 304, 16789, 5671, 315, 3738, 8231, 13, 63797, 41578, 315, 43216, 2142, 2924, 8185, 429, 432, 374, 32910, 38929, 11, 16387, 11, 476, 8620, 47354, 13, 31725, 97246, 11, 56245, 11, 323, 7271, 4710, 785, 1842, 1600, 5729, 6238, 315, 43216, 2142, 374, 504, 279, 36758, 17834, 458, 838, 71, 685, 11, 7290, 330, 28891, 264, 47678, 497, 23350, 315, 279, 9249, 458, 12, 3489, 28891, 899, 323, 279, 3409, 796, 30544, 436, 3489, 37197, 1, 476, 330, 81, 8478, 1827, 785, 20482, 481, 2142, 70577, 279, 41597, 1482, 429, 9241, 2471, 458, 15260, 13, 2082, 1113, 2142, 7951, 304, 6364, 504, 220, 126293, 17, 438, 43216, 43269, 323, 458, 15260, 504, 220, 122876, 24, 26, 4124, 6364, 601, 1134, 20014, 4056, 264, 5530, 315, 19231, 13]))
print(tokenizer.decode( [99638, 99900, 99254, 99058, 100022, 100047, 101580, 123934, 3837, 100566, 98404, 98924, 100433, 98582, 99156, 116897, 99065, 115036, 100918, 101580, 99291, 1773, 124795, 3837, 99392, 100918, 101580, 126404, 108636, 5373, 99705, 101117, 101268, 3837, 99281, 98582, 99900, 116897, 99065, 99322, 100918, 101580, 108277, 100566, 98404, 101580, 102313, 98801, 1773, 118160, 3837, 101580, 102313, 98582, 100918, 101580, 99291, 3837, 98396, 116897, 99065, 99322, 100918, 101580, 108277, 113580, 1773, 124795, 3837, 99270, 109579, 116897, 3837, 99013, 99483, 99665, 98738, 101580, 98548, 98341, 100832, 101580, 100433, 1773, 99928, 98853, 99135, 99705, 3837, 116897, 99065, 98316, 99486, 106631, 3837, 99630, 99205, 98346, 99900, 101580, 113316, 5373, 100832, 98458, 99254, 98413, 105329, 122680, 105325, 3837, 99348, 99205, 98346, 99528, 100652, 98638, 102101, 105325, 1773, 98319, 113225, 98322, 3837, 116897, 99065, 104080, 107951, 98316, 99116, 101919, 99730, 101119, 98622, 1773, 99900, 116897, 99065, 108279, 99665, 107123, 99928, 124375, 103765, 107951, 106786, 5373, 113579, 123542, 3837, 98316, 105004, 101281, 3837, 99002, 99379, 98314, 1773, 98487, 99068, 99486, 98378, 98942, 99900, 116897, 99065, 3837, 98359, 99630, 101077, 98435, 101580, 117864, 100832, 99355, 3837, 99348, 101077, 98435, 101429, 99900, 100225, 104043, 99355, 1773, 99900, 116897, 104401, 107951, 107228, 102737, 26852, 124375, 24991, 103765, 3837, 98811, 99900, 116897, 99065, 98857, 100836, 111056, 98333, 116813, 115568, 111190, 103655, 98394, 99065, 1773]))
print(tokenizer.vocab_size)