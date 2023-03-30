*INTRO*   
Deep Learning의 발전 과정을 간략히 정리 

<br>
<br>

__목차__  
1. AlexNet
2. DQN
3. Encoder/Decoder
4. Adam Optimizer
5. GAN
6. Residual Network
7. Transformer
8. BERT
9. BIG Language Models
10. Self Suprevised Learning


## 1. AlexNet
___
AlexNet은 ImageNet 데이터셋을 분류하는 문제를 푸는 모델로 DL연구의 붐을 일으킨 모델이다.  
AlexNet은 처음으로 일반화 기능을 향상시키는 dropout 방법이 처음으로 이 모델에 사용되었다.  
또한 AlexNet에서 사용된 Relu, convolutional layers, max-pooling은 컴퓨터비전에 기준이 되었다. 



## 2. DQN
___
Atari Games을 할 수 있는 모델로 어떤 게임을 하는지 알려주지 않아도 여러 게임을 알아서 할 수 있는 모델이다.  
이 모델은 강화학습을 통해 만들어졌다.  



## 3. Encoder, Decoder 
___
Atari Games을 할 수 있는 모델로 어떤 게임을 하는지 알려주지 않아도 여러 게임을 알아서 할 수 있는 모델이다.  
이 모델은 강화학습을 통해 만들어졌다. 


## 4. Adam Optimizer 
___





```python
import torch 
import numpy as np 

data = [1,2,3,4]

# ndarray to tensor
np_data = np.ndarray(data)
t_1 = torch.tensor(np_data)

# list to tensor
t_2 = torch.tensor(data)
```

Tensor와 Numpy와 다른 점이 있다면 Tensor는 `GPU`를 사용하여 계산할 수 있다는 것.

```python
t = torch.tensor(np.data)

# gpu 사용이 가능하다면 cuda로 바꿀 수 있다. 
if torch.cuda.is_available():
    t = t.to('cuda')
```

## 1. Tensor 선언 (Creation Ops)
---



## 2. Tensor의 모양 바꾸기 
___

`view` : tensor의 모양이 변형된 View tensor를 반환한다. 기존 tensor가 바뀌면 해당 view tensor의 값도 변한다.  

`reshape`: : tensor의 모양이 변형된 tensor를 반한한다. 기본 tensor가 바뀌더라도 반환받은 tensor는 변하지 않는다. 

[squeeze](https://pytorch.org/docs/stable/generated/torch.squeeze.html#torch.squeeze) : 1로 되어있는 tensor의 차원을 없앤다. 

[unsqueeze](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze) : tensor의 차원에 1을 추가한다. (2,2) -> (1,2,2)

## 3. Tensor의 연산 (Tensor Operation)
___

tensor간의 연산은 기본적으로 tensor의 모양이 같아야 연산을 수행할 수 있다. 

tensor간 곱셈 연산은 3가지가 있다. 

1. [dot](https://pytorch.org/docs/stable/generated/torch.dot.html#torch.dot): scalar 혹은 vector간의 연산을 지원한다. 2차원 이상의 tensor간의 곱셈은 지원하지 않는다.  

2. [mm](https://pytorch.org/docs/stable/generated/torch.mm.html#torch.mm) : matrix간의 곱셈만 지원한다.       broadcasting을 지원하지 않는다.

3. [matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul) : 다차원 tensor간의 곱셈을 지원한다. broadcasting도 지원한다. 

## Ref
---
https://dennybritz.com/posts/deep-learning-ideas-that-stood-the-test-of-time/ 