# RNN

*INTRO*   
Sequential RNN을 통해 

<br>    

__목차__  
1. Sequential 모델  
2. RNN
3. LSTM
4. GRU

<br>  

## 1. Sequential 모델
___

Sequential 모델은 처리하고자 하는 데이터에 일련의 순서 또는 시간의 개념이 존재하는 경우 사용되는 모델이다.  


1. __Naive Sequential Model__ : 이전 과거의 모든 정보를 이용하여 결과값을 내는 모델. 
2. __Autoegressive Model__ : 고정된 개수의 과거의 정보만을 이용하여 결과값을 내는 모델. 따라서 이 모델은 과거의 많은 정보를 버릴 수 밖에 없다.  
3. __Markov Model__ : 바로 이전 과거의 정보만이 현재 정보에 영향을 준다고 가정하고 바로 이전 과거의 정보만을 이용하여 결과값을 내는 모델.  
4. __Latent Autoregressive Model__ : 과거의 정보를 요약한 정보와 입력값을 이용하여 결과값을 내는 모델  

<br>  

## 2. RNN
___
RNN 모델은 자기 자신으로 돌아오는 구조가 있는 모델이다.  
자신에게 돌아오는 구조를 만든 이유는 이전의 정보를 계속해서 살려두고 이를 다음 값이 들어왔을 때 반영하게 하기 위함이다.  


단점
- vanishing gradient, exploding gradient 문제가 생길 수 있다.  
- 파라미터의 값이 계속해서 업데이트 되면서 이전에 있었던 정보가 살아남기 어렵다.  

## 3. LSTM
___
RNN의 모델이 Short term memory만을 고려하지 못하는 구조적인 문제를 해결하고자 나온 모델이다.   

LSTM은 cell state를 도입하여 위 문제를 해결했다. --> (왜?)  

cell state는 forget gate와 input gate를 통해 update한다.  

previous hidden state와 input, new cell state를 합쳐서 다음 next hidden state를 만든다.  

## 4. GRU
___


## Ref
---