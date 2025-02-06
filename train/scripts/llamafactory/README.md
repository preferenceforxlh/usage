# llamafactory 注意事项
* lora微调可以使用多个adpter，训练完成之后一起merge，Qlora只能指定一个adapter
* full微调不能指定adapter
  
# llamafactory train pipeline
* 加载tokenizer processor
    * 添加special token
    * patch 一些config
* 加载template
* 加载数据集
* 加载模型
    * 加载模型的配置文件，并且进行patch，主要针对attn实现方式,修改rope,配置量化方式
    * 加载模型，并且进行patch，主要针对是否需要resize vocab embedding,以及Qlora需要使用prepare_model_for_kbit_training进行包装
    * 初始化adapter，如果指定adapter_name，首先将adapter load到模型上，然后在创建新的adapter，应用到指定的模块上，如果模型resize vocab 也会应用在input embedding和output embedding上
* 创建data collator
* 创建trainer进行训练

## pt训练
### 数据集处理
* align转换成如下格式
```json
{
    "_prompt":[
        {"role":"user","content":""},
    ],
}
```
* process转换操作
将text + eos_token通过tokenizer转换成input_ids 

## sft训练
### 数据集处理
* align转换成如下格式
```json
{
    "_prompt":[
        {"role":"user","content":""},
        {"role":"assistant","content":""}
        {"role":"user","content":""},
    ],
    "_response":[
        {"role":"assistant","content":""}
    ]
}
```
* process转换操作
input_ids : query1 + resp1 + query2 + resp2 ...
labels :    IGNORE_INDEX * len(query1) + resp1_ids + IGNORE_INDEX * len(query2) + resp2_ids

## rm训练
### 数据集处理
* align转换成如下格式
```json
{
    "_prompt":[
        {"role":"user","content":""},
    ],
    "_response":[
        {"role":"assistant","content":"chosen"},
        {"role":"assistant","content":"rejected"}
    ]
}
```
* process转换操作
chosen_input_ids : promot_ids + chosen_ids 
chosen_labels : IGNORE_INDEX * len(prompt_ids) + chosen_ids
rejected_input_ids : prompt_ids + rejected_ids
rejected_labels : IGNORE_INDEX * len(prompt_ids) + rejected_ids

## ppo训练
### 数据集处理
* align转换成如下格式
```json
{
    "_prompt":[
        {"role":"user","content":""},
    ],
}
```
* process转换操作
input_ids : query_ids

### 加载reward模型和ref模型的方式
#### ref模型
* 如果指定ref model，那么会加载指定的ref model以及对应的adapter(如有)
* 如果是lora训练方式，那么ref model 为none
* 如果没指定ref model 并且训练方式不是lora 那么 ref model为sft model
### reward模型
* 如果奖励模型是api模型，直接使用远程api模型
* 如果奖励模型是lora类型，直接在sft模型上加adapter -- 因为通常奖励模型是sft模型训练的到的
* 如果奖励模型是full类型，直接加载指定reward model以及对应的adapter(如有)

## dpo训练
### 数据集处理
数据集处理方式和rm训练处理方式相同
### 加载ref模型的方式
* 如果指定ref model，那么会加载指定的ref model以及对应的adapter(如有)
* 如果是lora训练方式，那么ref model 为none
* 如果没指定ref model 并且训练方式不是lora 那么 ref model为sft model