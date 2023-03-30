*INTRO*   
Deep Learning의 발전 과정을 간략히 정리 

<br>
<br>

__목차__  
1. 단층 퍼셉트론  
2. 다층 퍼셉트론
3. 신경망



## 1. 단층 퍼셉트론과 다층 퍼셉트론
___
단층 퍼셉트론을 통해 여러가지 논리 회로를 구성할 수 있다.  
그러나 단층 퍼셉트론 만으로는 XOR과 같은 논리 회로는 구성할 수 없다. 

퍼셉트론 여러로 구성된 다층 퍼셉트론을 통해 단층 퍼셉트론의 한계를 극복할 수 있다.  


## 2. 신경망
___





## 3. 신경망
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