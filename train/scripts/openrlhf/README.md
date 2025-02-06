# tips for OpenRLHF

## pretrain
**数据集形式**:
```json
{
"input_key":"text"
}
```
* 数据格式:text + eos_token
* unpacked:将每条数据的input_ids padding起来
  - 模型输入:[bs,seq_len] 
* packed:将多条数据合并成一条?是否回超过max_length?
  - 模型输入:[1,all_seq_len]
* loss:GPT loss 
* tokenizer_pad_side:right
## sft
**数据集形式**
```json
(推荐)
{
    "input_key":[
        {"role":"user","content":""},
        {"role":"assistant","content":""},
    ]
}
(推荐，类似于alpaca，input_key = instruction + input)
{
    "input_key":"",
    "output_key":""
}
```
```json
{
    "input_key":[
        {"role":"user","content":""},
        {"role":"assistant","content":""},
        {"role":"user","content":""},
    ],
    "output_key":[
        {"role":"assistant","content":""}
    ]
}
```
* 数据格式：chat-template
* 多轮对话数据集仅仅学习最后一轮对话
* unpacked:将每条数据的input_ids padding起来
  - 模型输入:[bs,seq_len] 
* packed:将多条数据合并成一条?是否回超过max_length?
  - 模型输入:[1,all_seq_len]
* loss:GPT loss 
* tokenizer_pad_side:right
## rm
**数据集形式**
```json
{
    "prompt":[
        {"role":"user","content":""}
    ],
    "chosen_key":[
        {"role":"assistant","content":""}
    ],
    "rejected_key":[
        {"role":"assistant","content":""}
    ]
}
```
```json
(推荐)
{
    "chosen_key":[
        {"role":"user","content":""},
        {"role":"assistant","content":""}
    ],
    "rejected_key":[
        {"role":"user","content":""},
        {"role":"assistant","content":""}
    ]
}
```
* 数据格式：chat-template
* unpacked:将每条数据的chosen_input_ids,rejected_input_ids padding起来
  - 模型输入:
    - chosen_input_ids -> [bs,seq_len] 
    - rejected_input_ids -> [bs,seq_len]
    - 之后将两者cat起来，all_input_ids = torch.cat([chosen_input_ids,rejected_input_ids],dim=0) -> [2 * bs,seq_len]
    - 奖励模型会利用每个文本最后一个token输出整个句子的奖励值，因为最后一个token聚合整个句子的信息
* packed:将多条数据chosen_input_ids和rejected_input_ids合并成一条?是否回超过max_length?
  - 模型输入:
    - all_input_ids -> [1,all_seq_len]
    - 奖励模型为每个token输出奖励值，之后利用seq_len去索引每个句子最后一个token获取每个句子的奖励分数
* loss:-logsigmod(chosen_score - rejected_score),score使用最后一个token输出value，作为整个句子的value值
* tokenizer_pad_side:left
## ppo
```json
{
    "prompt":[
        {"role":"user","content":""}
    ]
}
```
```json
(推荐)
{
    "input_key":""
}
```
* 如果奖励模型不是远程,Critic模型来自奖励模型,否则奖励模型来自sft模型,Ref标准模型来自Actor模型
* ppo流程
  - 制作expecience
    - 从prompt数据集中获取一组prompt
    - 传入到LLM进行生成一组sample
    - 将sample送入actor模型计算action的logprob
    - 将sample送入init模型计算action的ref logprob
    - 将sample送入critic模型计算，每个状态对应的状态价值
    - 将sample送入奖励模型输出每个句子的奖励分数
    - 计算logprob和ref logprob的kl散度，保证actor模型保持原有的ref模型的token选取风格
    - 计算总的奖励分数:-kl + 整个句子的奖励分数，只加在最后一个token,最终得到每个action的奖励
    - 计算advantage -- t及时奖励 + gamma * t+1状态奖励  - t状态奖励 + lambda * gamma * t+1优势 和 return -- advantage + 对应状态价值
  - 计算actor loss -- clip之后的advantage 以及 critic loss -- MSE(预测状态价值，return)
  - 更新模型参数

## dpo
* 直接利用偏好数据集计算loss
* 只使用actor模型和ref模型