---
license: mit
base_model:
- moonshotai/Moonlight-16B-A3B
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- agent
- video
- screenspot
- long-context
---



<div align="center">
  <img width="30%" src="figures/logo.png">
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2504.07491">
    <b>📄 Tech Report</b>
  </a> &nbsp;|&nbsp;
  <a href="https://github.com/MoonshotAI/Kimi-VL">
    <b>📄 Github</b>
  </a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/moonshotai/Kimi-VL-A3B/">💬 Chat Web</a>
</div>


## Introduction

We present **Kimi-VL**, an efficient open-source Mixture-of-Experts (MoE) vision-language model (VLM) that offers **advanced multimodal reasoning, long-context understanding, and strong agent capabilities**—all while activating only **2.8B** parameters in its language decoder (Kimi-VL-A3B).

Kimi-VL demonstrates strong performance across challenging domains:
as a general-purpose VLM, Kimi-VL excels in multi-turn agent interaction tasks (e.g.,OSWorld), achieving state-of-the-art results comparable to flagship models.
Furthermore, it exhibits remarkable capabilities across diverse challenging vision language tasks, including college-level image and video comprehension, optical character recognition (OCR), mathematical reasoning, multi-image understanding, and etc.

In comparative evaluations, it effectively competes with cutting-edge efficient VLMs such as GPT-4o-mini, Qwen2.5-VL-7B, and Gemma-3-12B-IT, while surpassing GPT-4o in several specialized domains.

Kimi-VL also advances the pareto frontiers of multimodal models in processing long contexts and perceiving clearly: Equipped with a 128K extended context window, Kimi-VL can processes long and diverse inputs, achieving impressive scores of 64.5 on LongVideoBench, and 35.1 on MMLongBench-Doc; Its native-resolution vision encoder, MoonViT, further allows it to see and understand ultra-high-resolution visual inputs, achieving 83.2 on InfoVQA and 34.5 on ScreenSpot-Pro, while maintaining lower computational cost with common visual inputs and general tasks.

Building on this foundation, we introduce an advanced long-thinking variant: **Kimi-VL-Thinking**. Developed through long chain-of-thought (CoT) supervised fine-tuning (SFT) and reinforcement learning (RL), this model exhibits strong long-horizon reasoning capabilities. It achieves scores of 61.7 on MMMU, 36.8 on MathVision, and 71.3 on MathVista while maintaining the compact 2.8B activated LLM parameter footprint, setting a new standard for efficient yet capable multimodal **thinking** models.

## Architecture

The model adopts an MoE language model, a native-resolution visual encoder (MoonViT), and an MLP projector, as illustrated in the following image.

<div align="center">
  <img width="90%" src="figures/arch.png">
</div>

## Model Variants

🤗 For general multimodal perception and understanding, OCR, long video and long document, video perception, and agent uses, we recommend `Kimi-VL-A3B-Instruct` for efficient inference; for advanced text and multimodal reasoning (e.g. math), please consider using `Kimi-VL-A3B-Thinking`.

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download Link** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| Kimi-VL-A3B-Instruct | 16B | 3B | 128K   | [🤗 Hugging Face](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)   |
| Kimi-VL-A3B-Thinking  | 16B | 3B |  128K   | [🤗 Hugging Face](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking)   |

</div>

> [!Note]
> Recommended parameter settings:
> - For **Thinking models**, it is recommended to use `Temperature = 0.8`. 
> - For **Instruct models**, it is recommended to use `Temperature = 0.2`.
> - Greedy sampling (`Temperature = 0.0`) is okay for non-thinking (instruct) models (aligned with our evaluation setting).

## Performance

As an efficient model, Kimi-VL can robustly handle diverse tasks (fine-grained perception, math, college-level problems, OCR, agent, etc) across a broad spectrum of input forms (single-image, multi-image, video, long-document, etc).


A brief comparison with existing 10B-level dense VLMs and DeepSeek-VL2 (A4.5B):

<div align="center">
  <img width="100%" src="figures/instruct_perf.png">
</div>

Full comparison (GPT-4o included for reference):

<div align="center">

| Benchmark (Metric)             | GPT-4o | GPT-4o-Mini | Qwen2.5-VL-7B | Llama3.2-11B-Inst. | Gemma3-12B-IT | DeepSeek-VL2 | Kimi-VL-A3B-Instruct |
|--------------------------------|--------|-------------|---------------|--------------------|---------------|--------------|-------------|
| **Architecture**               | -      | -           | Dense         | Dense              | Dense         | MoE          | MoE         |
| **# Act. Params (LLM+VT)**     | -      | -           | 7.6B+0.7B     | 8B+2.6B            | 12B+0.4B      | 4.1B+0.4B    | 2.8B+0.4B   |
| **# Total Params**             | -      | -           | 8B            | 11B                | 12B           | 28B          | 16B         |
|                                |        |             |               |                    |               |              |             |
| **College-level**              |        |             |               |                    |               |              |             |
| MMMU-Val (Pass@1)                | *69.1*   | **60.0**    | 58.6          | 48                 | 59.6          | 51.1         | 57.0        |
| VideoMMMU (Pass@1)              | *61.2*   | -           | 47.4          | 41.8               | **57.2**      | 44.4         | 52.6        |
| MMVU-Val (Pass@1)               | *67.4*   | **61.6**    | 50.1          | 44.4               | 57.0          | 52.1         | 52.2        |
|                                |        |             |               |                    |               |              |             |
| **General**                    |        |             |               |                    |               |              |             |
| MMBench-EN-v1.1 (Acc)          | *83.1*   | 77.1        | 82.6          | 65.8               | 74.6          | 79.6         | **83.1**    |
| MMStar (Acc)                   | *64.7*   | 54.8        | **63.9**      | 49.8               | 56.1          | 55.5         | 61.3        |
| MMVet (Pass@1)                 | *69.1*   | 66.9        | **67.1**      | 57.6               | 64.9          | 60.0         | 66.7        |
| RealWorldQA (Acc)              | *75.4*   | 67.1        | **68.5**      | 63.3               | 59.1          | 68.4         | 68.1        |
| AI2D (Acc)                     | *84.6*   | 77.8        | 83.9          | 77.3               | 78.1          | 81.4         | **84.9**    |
|                                |        |             |               |                    |               |              |             |
| **Multi-image**                |        |             |               |                    |               |              |             |
| BLINK (Acc)                    | *68.0*   | 53.6        | 56.4          | 39.8               | 50.3          | -            | **57.3**    |
|                                |        |             |               |                    |               |              |             |
| **Math**                       |        |             |               |                    |               |              |             |
| MathVista (Pass@1)             | *63.8*   | 52.5        | 68.2          | 47.7               | 56.1          | 62.8         | **68.7**    |
| MathVision (Pass@1)            | *30.4*   | -           | 25.1          | 13.6               | **32.1**      | 17.3         | 21.4        |
|                                |        |             |               |                    |               |              |             |
| **OCR**                        |        |             |               |                    |               |              |             |
| InfoVQA (Acc)                  | *80.7*   | 57.9        | 82.6          | 34.6               | 43.8          | 78.1         | **83.2**    |
| OCRBench (Acc)                 | *815*    | 785         | 864           | 753                | 702           | 811          | **867**     |
|                                |        |             |               |                    |               |              |             |
| **OS Agent**                   |        |             |               |                    |               |              |             |
| ScreenSpot-V2 (Acc)            | *18.1*   | 6.9         | 84.2          | -                  | -             | -            | **92.8**    |
| ScreenSpot-Pro (Acc)           | *0.8*    | -           | 29.0          | -                  | -             | -            | **34.5**    |
| OSWorld (Pass@1)               | *5.03*   | -           | 2.5           | -                  | -             | -            | **8.22**    |
| WindowsAgentArena (Pass@1)     | *9.4*    | 2.7         | 3.4           | -                  | -             | -            | **10.4**    |
|                                |        |             |               |                    |               |              |             |
| **Long Document**              |        |             |               |                    |               |              |             |
| MMLongBench-Doc (Acc)          | *42.8*   | 29.0        | 29.6          | 13.8               | 21.3          | -            | **35.1**    |
|                                |        |             |               |                    |               |              |             |
| **Long Video**                 |        |             |               |                    |               |              |             |
| Video-MME (w/o sub.)     | *71.9* | 64.8 | 65.1    | 46.0         | 58.2     | -            | **67.8** |
| Video-MME (w sub.)     | *77.2* | 68.9 | 71.6    | 49.5          | 62.1     | -            | **72.6** |
| MLVU-MCQ (Acc)                  | *64.6*   | 48.1        | 70.2          | 44.4               | 52.3          | -            | **74.2**    |
| LongVideoBench (val)           | *66.7*   | 58.2        | 56.0          | 45.5               | 51.5          | -            | **64.5**    |
|                                |        |             |               |                    |               |              |             |
| **Video Perception**           |        |             |               |                    |               |              |             |
| EgoSchema (full)               | 72.2   | -           | 65.0          | 54.3               | 56.9          | 38.5         | **78.5**    |
| VSI-Bench                      | 34.0   | -           | 34.2          | 20.6               | 32.4          | 21.7         | **37.4**    |
| TOMATO                         | *37.7*   | 28.8        | 27.6          | 21.5               | 28.6          | 27.2         | **31.7**    |

</div>

### Inference with 🤗 Hugging Face Transformers 

> [!Note]
> Recommended prompt for OS agent tasks (Expected output is a point):
> - `Please observe the screenshot, please locate the following elements with action and point.<instruction> [YOUR INSTRUCTION]`

We introduce how to use our model at inference stage using transformers library. It is recommended to use python=3.10, torch>=2.1.0, and transformers=4.48.2 as the development environment. 

```python
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "moonshotai/Kimi-VL-A3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image_path = "./figures/demo.png"
image = Image.open(image_path)
messages = [
    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": "What is the dome building in the picture? Think step by step."}]}
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
response = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)
```

### Inference with VLLM

We have submitted a Merge Request [#16387](https://github.com/vllm-project/vllm/pull/16387) to vLLM. You are welcome to deploy Kimi-VL using the branch corresponding to the vLLM MR until the MR is merged.

## Citation

```
@misc{kimiteam2025kimivltechnicalreport,
      title={{Kimi-VL} Technical Report}, 
      author={Kimi Team and Angang Du and Bohong Yin and Bowei Xing and Bowen Qu and Bowen Wang and Cheng Chen and Chenlin Zhang and Chenzhuang Du and Chu Wei and Congcong Wang and Dehao Zhang and Dikang Du and Dongliang Wang and Enming Yuan and Enzhe Lu and Fang Li and Flood Sung and Guangda Wei and Guokun Lai and Han Zhu and Hao Ding and Hao Hu and Hao Yang and Hao Zhang and Haoning Wu and Haotian Yao and Haoyu Lu and Heng Wang and Hongcheng Gao and Huabin Zheng and Jiaming Li and Jianlin Su and Jianzhou Wang and Jiaqi Deng and Jiezhong Qiu and Jin Xie and Jinhong Wang and Jingyuan Liu and Junjie Yan and Kun Ouyang and Liang Chen and Lin Sui and Longhui Yu and Mengfan Dong and Mengnan Dong and Nuo Xu and Pengyu Cheng and Qizheng Gu and Runjie Zhou and Shaowei Liu and Sihan Cao and Tao Yu and Tianhui Song and Tongtong Bai and Wei Song and Weiran He and Weixiao Huang and Weixin Xu and Xiaokun Yuan and Xingcheng Yao and Xingzhe Wu and Xinxing Zu and Xinyu Zhou and Xinyuan Wang and Y. Charles and Yan Zhong and Yang Li and Yangyang Hu and Yanru Chen and Yejie Wang and Yibo Liu and Yibo Miao and Yidao Qin and Yimin Chen and Yiping Bao and Yiqin Wang and Yongsheng Kang and Yuanxin Liu and Yulun Du and Yuxin Wu and Yuzhi Wang and Yuzi Yan and Zaida Zhou and Zhaowei Li and Zhejun Jiang and Zheng Zhang and Zhilin Yang and Zhiqi Huang and Zihao Huang and Zijia Zhao and Ziwei Chen},
      year={2025},
      eprint={2504.07491},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.07491}, 
}
```