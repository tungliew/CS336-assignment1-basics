# Stanford CS336 Assignment1 Spring 2026
<br>

## 所有functions测试结果
46 passed, 2 skipped
![CS336](./results/pass_1.png)
<br>
![CS336](./results/pass_2.png)

## BPETokenizer 训练结果
1. 在TinyStories数据集上进行训练
2. vocab_size = 10,000
3. 训练时长 大约35min

<br>

## TransformerLM 模型训练结果
1. 同样在TinyStories数据集上进行训练
2. 全程CPU训练，训练时长大约3h
   (因为中间使用过GPT-2 tokenizer做过测试，训练时忘记把vocab_size改回10,000了，直接用vocab_size=50,257进行的训练，除了参数变多其他没有影响）
4. 资源内容统计
================================================================================
MODEL CONFIG
================================================================================
vocab_size      : 50,257
context_length  : 256
num_layers      : 4
d_model         : 512
num_heads       : 16
d_ff            : 1,344


================================================================================
PARAMETERS
================================================================================
Total parameters: 63,919,616


================================================================================
MEMORY
================================================================================
Memory (bytes): 255,678,464
Memory (MB):    243.83
Memory (GB):    0.24


================================================================================
FORWARD PASS FLOPs
================================================================================
FLOPs per layer: 1,728,053,248
Transformer FLOPs: 6,912,212,992
LM head FLOPs: 13,174,571,008


Total forward FLOPs: 20,086,784,000
Total TFLOPs: 0.020

5. 训练结果loss的变化
  ![CS336](./results/loss_charg.png) 

## 使用训练后模型生成文本
prompt = "The dragon opened the door"
生成文本如下：
The dragon opened the door. The bird said, "I love you for the bird. What fly."
"Maybe you have a small forest!" laughed and Lily. Tom said, "That's you sad, Lily. I don'
"I am sad and play with you," was so much.
One day, Tom can the heavy tree.
"No, you are not many boat's, and be careful. Tim's okay, I want to be careful on the good
Tom's mom went to the bunny and the dog and the car every day.
"I will go. You don't find his friends."
"OK, I'm proud of the ball's friend. She wanted to be toys. You can play together. He foun
The dog was scared. She decided to give her friends. The boy saw the tree and was so tired
<|endoftext|>
