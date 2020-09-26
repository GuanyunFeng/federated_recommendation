# federated_recommendation
使用tensorflow2.1实现联合学习推荐模型，并加入差分隐私噪声进行隐私保护。</br>
推荐算法参考论文：[Neural Collaborative Filtering](https://dl.acm.org/doi/10.1145/3038912.3052569)
## 运行环境：
  python==3.7</br>
  numpy==1.18.1</br>
  tensorflow==2.1.0</br>
  scipy==1.4.1
## 使用方式
  非联合学习模式：</br>
  ```python single.py```</br>
  联合学习模式：</br>
  ```python server.py```
