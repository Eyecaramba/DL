*INTRO*   
Deep Learning에서 중요한 여러가지 Optimization 개념들을 정리

<br>
<br>

__목차__  
1. Optimization에서 자주 등장하는 개념들  
2. Gradient Descent Methods
3. Regularization 



## 1. model을 이루는 중요한 요소들 
___

Data : 모델을 학습시키기 위한 데이터.  
Model : 데이터를 변형한다.  
Loss : 좋고 나쁨을 측정할 수 있게 한다. 
Algorithm : Loss를 최소화 할 수 있는 알고리즘 (ex 경사하강법, dropout, early stopping)


Loss는 우리가 원하는 것의 근사치로 볼 수 있다. 따라서 원하는 것의 따라서 loss의 정의는 달라져야 한다.  

mse : 
binary corss entropy loss 
categorical cross entropy loss 
hinge loss 
kl-divergence loss 





1. __Generalization__ : 모델이 훈련 데이터에서 학습할 패턴을 새로운 데이터에 일반화할 수 있는 능력. 특정 데이터에만 맞춰저 있는 것이 아니라 모델이 이전에 본 적 없는 데이터에서도 좋은 성능을 잘 낼 수 있는 능력을 의미한다. Deep Learnig 모델이 추구하는 목표이기도 하다.  
(overfitting과 underfitting은 genearlization의 반대에 해당하는 의미이다. )
2. __Cross Validation__ : 모델을 더 일반화시키기 위해 데이터를 여러개의 다른 부분집합으로 나누어 테스트셋으로 사용하는 것을 말한다.  
(위 개념을 실현하는 방식으로 holdout method, k-fold cross validation, leave -p out cross validation 등 여러가지 방법이 있다.) 
3. __Bias-variance tradeoff__ : 편향과 분산이 tradeoff 관계가 있음을 의미한다. 따라서 두 값을 동시에 줄일 수 없음을 암시한다.  
4. __Boostraping__ : 데이터셋에서 무작위로 데이터를 선정하고 다시 복구시키는 방법을 반복하여 다른 데이터 조합을 가진 데이터셋을 여러개 만드는 것.    
5. __Bagging__:  boostraping으로 만든 여러 모델들이 생성한 예측 결과를 결합하여 단일 예측 결과를 생성하는 방법.
6. __Boosting__ : 모델이 잘 예측하지 못하는 데이터만을 따로 모아 다시 학습시키고 이를 이전 모델에 붙이는 방법.   

## 2. 경사하강법의 종류와 일반화 
___
경사하강법에는 여러가지 방법이 존재한다.  
같은 모델에 같은 데이터라고 하더라도 경사하강법에 따라 모델은 일반화가 잘 되어 있기도 하고 그렇지 않기도 하다.  


첫번째로 경사하강법은 배치 사이즈에 따라 종류를 나눌 수 있다. 
1. __SGD__ : 무작위로 뽑은 하나의 데이터를 사용하는 방식  
2. __mini batch gradient descent__ : 랜덤하게 뽑은 mini batch 단위를 사용하는 방식
3. __batch gradient descent__ : 데이터 전체를 한번에 사용하는 방식

> 일반적으로 작은 batch를 사용할수록 일반화 성능이 높아지고 batch가 클수록 일반화 성능이 낮아지는 것으로 알려져 있다.  
on large-batch training for deep learning : generalization gap and sharp minima
 

경사하강법은 배치 사이즈 뿐만 아니라 파라미터 업데이트 방식에 따라서도 여러가지 방법이 있다.  
1. SGD
- 장점 : 
- 단점 : 
2. Momentum :  
3. Nesterov accelerated gradient : 
4. Adagrad : 
5. Adadelta : 
6. RMSprop : 
7. Adam :  



어떤 경사 하강법을 쓰는가에 따라 같은 모델에 같은 데이터를 사용한다고 하더라도 overfitting또는 underfitting이 될 수 있고 generalization이 될 수 도 있다.  


## 3. Regularization 
___
모델의 overfitting을 막기 위한 여러가지 방법들이 존재한다.  

1. Early Stopping : vaildation set의 오차가 줄어들다가 다시 증가하는 시점을 찾고 이 시점에서 학습을 중단시키는 것.  

2. Parameter Norm Penalty : 

3. Data Augmentation : 


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