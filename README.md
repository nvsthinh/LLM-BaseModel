# LLM-BaseModel
## 🔥 Base Model
- Llama [[paper](https://arxiv.org/pdf/2302.13971.pdf)] [[code](https://github.com/facebookresearch/llama)] [[model](https://huggingface.co/meta-llama)]
  - 2023/02, Meta AI proposes the open source LLM Llama, which has four scales: 7b, 13b, 33b, and 65b.

- ChatGLM [[paper](https://arxiv.org/pdf/2103.10360.pdf)] [[code](https://github.com/THUDM/ChatGLM-6B/blob/main/README.md)] [[model](https://huggingface.co/THUDM/chatglm-6b)]
  - 2023/03, Tsinghua University proposes the open bilingual language model ChatGLM, based on [General Language Model](https://github.com/THUDM/GLM) framework, with the specification of 7b.

- Alpaca [[paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[code](https://github.com/tatsu-lab/stanford_alpaca)] [[model](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff/tree/main)]

  - 2023/03, Stanford University proposes Alpaca, an open source LLM fine-tuned based on the Llama 7b model. There are 1 specification of 7b, and the training is simpler and cheaper.

- Vicuna [[paper](https://lmsys.org/blog/2023-03-30-vicuna/)] [[code](https://github.com/lm-sys/FastChat)] [[model](https://huggingface.co/lmsys)]
  - 2023/03, UC Berkeley University, CMU and Stanford University propose Vicuna, an open souce LLM based on the  Llama model, has two specifications: 7b and 13b.

- WizardLM [[paper](https://arxiv.org/pdf/2304.12244.pdf)] [[code](https://github.com/nlpxucan/WizardLM)] [[model](https://huggingface.co/WizardLM)]
  - 2023/04, Peking University and Microsoft propose WizardLM, a LLM of evolutionary instructions, with three specifications of 7b, 13b, and 30b. 2023/06, They propose WizardMath, a LLM in the field of mathematics. 2023/08, They propose WizardCoder, a LLM in the field of code.

- Falcon [[paper](https://arxiv.org/pdf/2306.01116.pdf)] [[code](https://huggingface.co/tiiuae/falcon-180B)] [[model](https://huggingface.co/tiiuae)]
  - 2023/06, United Arab Emirates proposes Falcon, an open source LLM trained solely on refinedweb datasets, with four parameter specifications of 1b, 7b, 40b and 180b. It is worth noting that the performance on model 40B exceeds that of 65B LLaMA. 
  
- ChatGLM2[[paper](https://arxiv.org/pdf/2210.02414.pdf)] [[code](https://github.com/THUDM/ChatGLM2-6B/blob/main/README_EN.md)] [[model](https://huggingface.co/THUDM/chatglm2-6b)]
  - 2023/06, Tsinghua University proposes the second-generation version of ChatGLM，with the specification of 7b, which has stronger performance, longer context, more efficient inference and more open license.

- Baichuan-7b [[code](https://github.com/baichuan-inc/baichuan-7B)] [[model](https://huggingface.co/baichuan-inc/Baichuan-7B)]
  - 2023/06, Baichuan Intelligent Technology proposes the Baichuan-7B, an open-source, large-scale pre-trained language model based on Transformer architecture, which contains 7 billion parameters and trained on approximately 1.2 trillion tokens. It supports both Chinese and English languages with a context window length of 4096. 
  
- Baichuan-13b [[code](https://github.com/baichuan-inc/Baichuan-13B)] [[model](https://huggingface.co/baichuan-inc/Baichuan-13B-Base)]
  - 2023/07, Baichuan Intelligent Technology proposes the Baichuan-13B, an open-source, commercially available large-scale language model, following Baichuan-7B, which has two versions: pre-training (Baichuan-13B-Base) and alignment (Baichuan-13B-Chat).

- InternLM [[paper](https://github.com/InternLM/InternLM-techreport/blob/main/InternLM.pdf)] [[code](https://github.com/InternLM/InternLM/)] [[model](https://huggingface.co/internlm)]
  - 2023/07, Shanghai AI Laboratory and SenseTime propose the InternLM,  which has open-sourced a 7b and 20b parameter base models and chat models tailored for practical scenarios and the training system.

- Llama 2 [[paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)] [[code](https://github.com/facebookresearch/llama)] [[model](https://huggingface.co/meta-llama)]
  - 2023/07, Meta AI proposes the second-generation Llama series open-source LLM Llama 2. Compared with Llama 1, the training data is 40% more, and the context length is doubled. The model has four specifications: 7b, 13b, 34b, and 70b, but 34b is not open source. 

- Code Llama [[paper](https://arxiv.org/pdf/2308.12950.pdf)] [[code](https://github.com/facebookresearch/codellama)] [[model](https://huggingface.co/codellama)]
  - 2023/08, Meta AI proposes Code LLama, based on Llama 2. Code Llama reaches state-of-the-art performance among open models on several code benchmarks. There are foundation models (Code Llama), Python specializations (Code Llama - Python), and instruction-following models,  with 7B, 13B and 34B parameters each. 2024/01, Meta AI open sourced CodeLlama-70b, CodeLlama-70b-Python and CodeLlama-70b-Instruct.

- Qwen [[paper](https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN_TECHNICAL_REPORT.pdf)] [[code](https://github.com/QwenLM/Qwen)] [[model](https://huggingface.co/Qwen)]
  - 2023/08, Alibaba Cloud proposes the 7b-parameter version of the large language model series Qwen-7B (abbr. Tongyi Qianwen), is pretrained on a large volume of data, including web texts, books, codes, etc, which has open sourced two models with Qwen-7B and Qwen-7B-Chat. 2023/09, Alibaba Cloud updated the Qwen-7B and Qwen-7B-Chat and open sourced Qwen-14B and Qwen-14B-Chat. 2023/11, they open sourced Qwen-1.8B, Qwen-1.8B-Chat, Qwen-72B and Qwen-72B-Chat.

- Baichuan 2 [[paper](https://arxiv.org/pdf/2309.10305.pdf)]
[[code](https://github.com/baichuan-inc/Baichuan2)] [[model](https://huggingface.co/baichuan-inc)]
  - 2023/09, Baichuan Intelligent Technology proposes the new generation of open-source large language models Baichuan 2, trained on a high-quality corpus with 2.6 trillion tokens, which has base and chat versions for 7B and 13B, and a 4bits quantized version for the chat model.

- Phi-1.5 [[paper](https://arxiv.org/pdf/2309.05463.pdf)] [[model](https://huggingface.co/microsoft/phi-1_5)]
  - 2023/09, Microsoft Research proposes the open source language model phi-1.5, a Transformer with 1.3 billion parameters, which was trained using the same data sources as [phi-1](https://huggingface.co/microsoft/phi-1), augmented with a new data source that consists of various NLP synthetic texts. When assessed against benchmarks testing common sense, language understanding, and logical reasoning, phi-1.5 demonstrates a nearly state-of-the-art performance among models with less than 10 billion parameters. 2023/12, They propose [Phi-2](https://huggingface.co/microsoft/phi-2), a 2.7 billion-parameter language model that demonstrates outstanding reasoning and language understanding capabilities, showcasing state-of-the-art performance among base language models with less than 13 billion parameters. 

 - Mistral-7B [[paper](https://arxiv.org/pdf/2310.06825.pdf)]
    [[code](https://github.com/mistralai/mistral-src)] 
    [[model](https://huggingface.co/mistralai/Mistral-7B-v0.1)]
   - 2023/10, Mistral-AI company proposes the open source LLM Mistral 7B, a 7–billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms the best open 13B model (Llama 2) across all evaluated benchmarks, and the best released 34B model (Llama 1) in reasoning, mathematics, and code generation. They also provide a model fine-tuned to follow instructions, Mistral 7B – Instruct, that surpasses Llama 2 13B–chat model both on human and automated benchmarks. 2023/12，They propose the open source LLM Mixtral-8x7B, a pretrained generative Sparse Mixture of Experts, which outperforms Llama 2 70B on most benchmarks.

 - Deepseek [[paper](https://arxiv.org/pdf/2401.02954.pdf)]
    [[code](https://github.com/deepseek-ai/DeepSeek-LLM)] 
    [[model](https://huggingface.co/deepseek-ai)]
   - 2023/11, DeepSeek-AI company proposes the open source LLM deepseek, which has been trained from scratch on a vast dataset of 2 trillion tokens in both English and Chinese. Similarly, the deepseek LLM mainly has two categories: base and chat, with two parameter formats of 7b and 67b respectively. Data from its paper shows that deepSeek LLM 67b surpasses LLaMA-2 70b across a range of benchmarks, especially in the domains of code, mathematics, and reasoning. Furthermore, DeepSeek LLM 67B Chat exhibits superior performance compared to GPT-3.5.

 - MiniCPM [[paper](https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a)]
    [[code](https://github.com/OpenBMB/MiniCPM)] 
    [[model](https://huggingface.co/openbmb)]
   - 2024/02, ModelBest Inc. and TsinghuaNLP proposes the open source LLM MiniCPM, which is an End-Side LLM, with only 2.4B parameters excluding embeddings (2.7B in total). It is worth that MiniCPM has very close performance compared with Mistral-7B on open-sourced general benchmarks with better ability on Chinese, Mathematics and Coding after SFT. The overall performance exceeds Llama2-13B, MPT-30B, Falcon-40B, etc.

 - Mixtral-8x22B [[paper](https://mistral.ai/news/mixtral-8x22b/)] [[code](https://docs.mistral.ai/getting-started/open_weight_models/)] [[model](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1)]
   - 2024/04, Mistral AI proposed the latest open model Mixtral 8x22B. It sets a new standard for performance and efficiency within the AI community. It is a sparse Mixture-of-Experts (SMoE) model that uses only 39B active parameters out of 141B, offering unparalleled cost efficiency for its size.

 - Phi-3 [[paper](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/)] [[model](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)]
   - 2024/04, Microsoft proposed the Phi-3 models, which are the most capable and cost-effective small language models (SLMs) available, outperforming models of the same size and next size up across a variety of language, reasoning, coding, and math benchmarks. Phi-3-mini is available in two context-length variants—4K and 128K tokens. It is the first model in its class to support a context window of up to 128K tokens, with little impact on quality. Phi-3-small (7B) and Phi-3-medium (14B) will be available in the Azure AI model catalog and other model gardens shortly.   

  - Llama 3 [[paper](https://ai.meta.com/blog/meta-llama-3/)] [[code](https://github.com/meta-llama/llama3)] [[model](https://huggingface.co/meta-llama)]
    - 2024/04, Meta AI proposed the third generation Llama series open source large model Llama 3. The model has 2 parameter specifications, 8b and 70b, with base and instruct versions respectively. Excitingly, Llama 3 models are a major leap over Llama 2 and establish a new state-of-the-art for LLM models at those scales.

  - Qwen-1.5-110B [[paper](https://qwenlm.github.io/blog/qwen1.5-110b/)] [[code](https://github.com/QwenLM/Qwen1.5)] [[model](https://huggingface.co/Qwen/Qwen1.5-110B)]
    - 2024/04, Alibaba Cloud proposed the first 100B+ model of the Qwen1.5 series, Qwen1.5-110B, which achieves comparable performance with Meta-Llama3-70B in the base model evaluation, and outstanding performance in the chat evaluation, including MT-Bench and AlpacaEval 2.0. Qwen1.5 is the beta version of Qwen2, which has 9 model sizes, including 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, and 110B dense models, and an MoE model of 14B with 2.7B activated.
  
  - Qwen2 [[paper](https://qwenlm.github.io/blog/qwen2/)] [[code](https://github.com/QwenLM/Qwen2)] [[model](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f)]
    - 2024/06, Alibaba Cloud proposed the evolution from Qwen1.5 to Qwen2, which has 5 model sizes, including Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, Qwen2-57B-A14B, and Qwen2-72B. Qwen2-72B exhibits superior performance compared to leading models such as Llama-3-70B. Notably, it surpasses the performance of its predecessor, Qwen1.5-110B, despite having fewer parameters.

  - Llama 3.1  [[paper](https://ai.meta.com/blog/meta-llama-3-1/)] [[code](https://github.com/meta-llama/llama3)] [[model](https://huggingface.co/meta-llama)]
    - 2024/07, Meta AI proposed the Llama 3.1 405B, which is the first openly available model that rivals the top AI models when it comes to state-of-the-art capabilities in general knowledge, steerability, math, tool use, and multilingual translation. As part of this latest release, they’re introducing upgraded versions of the 8B and 70B models. These are multilingual and have a significantly longer context length of 128K, state-of-the-art tool use, and overall stronger reasoning capabilities. 

  - Qwen2.5 [[paper](https://arxiv.org/abs/2407.10671)] [[code](https://github.com/QwenLM/Qwen2.5)] [[model](https://huggingface.co/Qwen)]
    - 2024/09, Alibaba Cloud proposed the latest addition to the Qwen family: Qwen2.5, along with specialized models for coding, Qwen2.5-Coder, and mathematics, Qwen2.5-Math. All open-weight models are dense, decoder-only language models, available in various sizes, including: Qwen2.5(0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B), Qwen2.5-Coder (1.5B, 7B, and 32B on the way) and Qwen2.5-Math (1.5B, 7B, and 72B). They benchmarked their largest open-source model, Qwen2.5-72B-Instruct against leading open-source models like Llama-3.1-70B-Instrct and Mistral-Large-V2-Instruct and achieved the best results in multiple indicators. 
    


 - Llama 3.2  [[paper](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)] [[code](https://github.com/meta-llama/llama3)] [[model](https://huggingface.co/meta-llama)]
    - 2024/09, Meta AI proposed the Llama 3.2, which includes small and medium-sized vision LLMs (11B and 90B), and lightweight, text-only models (1B and 3B) that fit onto edge and mobile devices, including pre-trained and instruction-tuned versions. The Llama 3.2 1B and 3B models support context length of 128K tokens and are state-of-the-art in their class for on-device use cases like summarization, instruction following, and rewriting tasks running locally at the edge. These models are enabled on day one for Qualcomm and MediaTek hardware and optimized for Arm processors.
    
    

 ## 💡 Fine-tuning
- P-Tuning [[paper](https://arxiv.org/pdf/2103.10385.pdf)] [[code](https://github.com/THUDM/P-tuning)] 
  - 2021/03, Tsinghua University and others propose P-Tuning, a fine-tuning method for LLM, which uses trainable continuous prompt word embeddings to reduce the cost of fine-tuning.

- LoRA [[paper](https://arxiv.org/pdf/2106.09685.pdf)] [[code](https://github.com/microsoft/LoRA)] 
  - 2021/06, Microsoft proposes the Low-Rank Adaptation method for fine-tuning LLM by freezing the pre-training weights.

- P-Tuning V2 [[paper](https://arxiv.org/pdf/2110.07602.pdf)] [[code](https://github.com/THUDM/P-tuning-v2)] 
  - 2021/10, Tsinghua University proposes P-Tuning V2, an improved version of P-Tuning with better performance.

- RLHF [[paper](https://huggingface.co/blog/rlhf)] [[code](https://github.com/huggingface/blog/blob/main/zh/rlhf.md)] 
  - 2022/12, OpenAI uses the RLHF (Reinforcement Learning from Human Feedback) method to train ChatGPT, and uses human feedback signals to directly optimize the language model, with excellent performance.

- RRHF [[paper](https://arxiv.org/pdf/2304.05302.pdf)] [[code](https://github.com/GanjinZero/RRHF)] 
  - 2023/04, Alibaba proposes a novel learning paradigm called RRHF（Rank Responses to Align Language Models
  with Human Feedback without tears), which can be tuned as easily as fine-tuning and achieve a similar
  performance as PPO in HH dataset.

- QLoRA [[paper](https://arxiv.org/pdf/2305.14314.pdf)] [[code](https://github.com/artidoro/qlora)] 
  - 2023/05, Washington University proposes the qlora method, based on the frozen 4bit quantization model, combined with LoRA method training, which further reduces the cost of fine-tuning.

- RLTF [[paper](https://arxiv.org/pdf/2307.04349.pdf)] [[code](https://github.com/Zyq-scut/RLTF)] 
  - 2023/07, Tencent proposes RLTF（Reinforcement Learning from Unit Test Feedback), a novel online RL framework with unit test feedback of multi-granularity for refining code LLMs.

- RRTF [[paper](https://arxiv.org/pdf/2307.14936v1.pdf)]
  - 2023/07, Huawei proposes RRTF（Rank Responses to align Test&Teacher Feedback). Compared with RLHF, RRHF can efficiently align the output probabilities of a language model with human preferences, with only 1-2 models required during the tuning period, and it is simpler than PPO in terms of implementation, hyperparameter tuning, and training.

- RLAIF [[paper](https://arxiv.org/pdf/2309.00267.pdf)]
  - 2023/09, Google proposes RLAIF (RL from AI Feedback), a technique where preferences are labeled by an off-the-shelf LLM in lieu of humans. They find that the RLHF and RLAIF methods achieve the similar results on the task of summarization.
