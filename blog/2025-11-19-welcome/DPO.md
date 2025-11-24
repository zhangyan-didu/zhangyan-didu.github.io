# DPO算法详解

#### Background：

**RLHF(Reinforcement Learning from Human Feedback)：**

流程：

首先由一组具有人类偏好的数据，然后使用极大似然估计的方法得到一个 **reward model**，reward model可以在给定 prompt 和大语言模型回答的情况下对大语言模型的回答进行打分。利用reward model进行强化学习。

#### DPO算法简介：

直接从用户偏好数据中进行学习，通过极大似然估计获得最终模型。

##### Benefit：

流程极大简化

#### 前置知识：

**KL散度：**

定义：

P 分布相对于 Q 分布的相似程度。KL散度的值大于等于0。P和Q越相似，KL散度越接近0 。如果P和Q分布完全一致，则KL散度等于0。

公式：
$$
KL(P||Q) = \sum_{x \in X}P(x)log(\frac{P(x)}{Q(x)})
$$
注释：

P相对于Q的KL散度，和 Q相对于P的KL散度，是不一样的。

关于为何KL散度大于等于0，直观理解：如果log部分是小于0的，则分子一定比较小，那么P(x)相对就偏小，而如果log部分大于0，则分子偏大，那么P(x)就偏大。这样乘以系数之后一定可以得到大于0的值。

**Bradley-Terry模型：**

功能：

对比较关系进行建模。例如，已知A与B的对战结果，A与C的对战结果，求B与C的对战预期结果。

定义：
$$
P(i>j) = \frac{\alpha_i}{\alpha_i + \alpha_j}
$$
假设每个函数都有一个隐含的实力参数，$\alpha_i$表示第 i 个元素的实力，$P(i>j)$表示第 i 个元素战胜第 j 个元素的概率。

我们可以通过对已知数据做**对数最大似然估计**得到实力参数的值。

据此定义一个一般的Loss函数，Loss越小越好：
$$
Loss=-\mathbb{E}_{(\alpha_x, \alpha_y)~D}[ln\frac{\alpha_x}{\alpha_x+\alpha_y}]
$$
这实际就是一个分类问题的交叉熵损失函数。

我们的优化目标就是让x（好回答）战胜y（坏回答）的概率趋近于1。

**强化学习里：**

输入的prompt是x，回答是y。回答y的好坏通过reward模型来评估。
$$
P(y_1>y_2)=\frac{r(x,y_1)}{r(x,y_1)+r(x,y_2)}
$$
r(x,y)有可能返回负数，所以加上指数函数
$$
P(y_1>y_2)=\frac{exp(r(x,y_1))}{exp(r(x,y_1)+r(x,y_2))}
$$
**Bradley_Terry模型**
$$
P(y_w>y_l)=\frac{exp(r(x,y_w))}{exp(r(x,y_w))+exp(r(x,y_l))}
$$

$$
\sigma(x)=\frac{1}{1+exp(-x)}
$$

$$
\begin{aligned}
Loss &= -\mathbb{E}_{(x,y_w,y_l)\sim D}\left[ln\frac{\exp(r(x,y_w))}{\exp(r(x,y_w))+\exp(r(x,y_l))}\right] \\
&= -\mathbb{E}_{(x,y_w,y_l)\sim D}[ln\frac{1}{1+exp(r(x,y_l)-r(x,y_w))}]\\
&=-\mathbb{E}_{(x,y_w,y_l)\sim D}[ln\sigma(r(x,y_w)-r(x,y_l))]
\end{aligned}
$$

$$
-ln\sigma(r(x,y_w)-r(x,y_l))
$$

让Loss尽可能的小，也就是让$y_w$通过reward方法的得分尽可能多于$y_l$的得分

#### DPO的训练目标

奖励函数：

$r(x,y)\ \ x:prompt\ \ y:response$

基准模型：

$\pi_{ref}(y|x)$

训练模型：

$\pi(y|x)$

**训练目标：**
$$
\underset{\pi}{max}\mathbb{E}_{x\sim D,y\sim\pi}[r(x,y)]-\beta\mathbb{D}_{KL}[\pi(y|x)||\pi_{ref}(y|x)]
$$
左半部分：得到尽可能多的奖励

右半部分：新训练的模型尽可能和基准模型分布一致。其中$\beta$是超参数，$\beta$越大，表示分布越应该一致
$$
\begin{aligned}
&\quad\ \underset{\pi}{max}\mathbb{E}_{x\sim D,y\sim\pi}[r(x,y)]-\beta\mathbb{D}_{KL}[\pi(y|x)||\pi_{ref}(y|x)]\\
&=\underset{\pi}{max}\mathbb{E}_{x\sim D,y\sim\pi}[r(x,y)]-\mathbb{E}_{x\sim D,y\sim\pi}[\beta log\frac{\pi(y|x)}{\pi_{ref}(y|x)}]\\
&=\underset{\pi}{max}\mathbb{E}_{x\sim D,y\sim\pi}[r(x,y)-\beta log\frac{\pi(y|x)}{\pi_{ref}(y|x)}]\\
&=\underset{\pi}{min}\mathbb{E}_{x\sim D,y\sim\pi}[log\frac{\pi(y|x)}{\pi_{ref}(y|x)}-\frac{1}{\beta}r(x,y)]
\end{aligned}
$$
（取$min$之后，两项换位置且两项均除以$\beta$）
$$
\begin{aligned}
&=\underset{\pi}{min}\mathbb{E}_{x\sim D,y\sim\pi}[log\frac{\pi(y|x)}{\pi_{ref}(y|x)}-\frac{1}{\beta}r(x,y)]\\
&=\underset{\pi}{min}\mathbb{E}_{x\sim D,y\sim\pi}[log\frac{\pi(y|x)}{\pi_{ref}(y|x)}-log\ exp\frac{1}{\beta}r(x,y)]\\
&=\underset{\pi}{min}\mathbb{E}_{x\sim D,y\sim\pi}[log\frac{\pi(y|x)}{\pi_{ref}(y|x)exp\frac{1}{\beta}r(x,y)}]\\
&=\underset{\pi}{min}\mathbb{E}_{x\sim D,y\sim\pi}[log\frac{\pi(y|x)}{\pi_{ref}(y|x)exp\frac{1}{\beta}r(x,y)\frac{1}{Z(x)}Z(x)}]\\
&=\underset{\pi}{min}\mathbb{E}_{x\sim D,y\sim\pi}[log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{ref}(y|x)exp\frac{1}{\beta}r(x,y)}-logZ(x)]\\
\end{aligned}
$$

$$
Z(x)=\sum_{y}\pi_{ref}(y|x)exp(\frac{1}{\beta}r(x,y))
$$

$$
{\frac{1}{Z(x)}\pi_{ref}(y|x)exp\frac{1}{\beta}r(x,y)}=\frac{\pi_{ref}(y|x)exp(\frac{1}{\beta}r(x,y))}{\sum_{y}\pi_{ref}(y|x)exp(\frac{1}{\beta}r(x,y))}=\pi^*(y|x)
$$

也就是对于一个x，「一种特定的y的概率」比上「所有y的概率之和」
$$
\begin{aligned}
&=\underset{\pi}{min}
\mathbb{E}_{x\sim D,y\sim\pi}[log\frac{\pi(y|x)}{\pi^*(y|x)}-logZ(x)]\\
&=\underset{\pi}{min}\mathbb{E}_{x\sim D,y\sim\pi}[log\frac{\pi(y|x)}{\pi^*(y|x)}]\\
&=\underset{\pi}{min}\mathbb{E}_{x\sim D}[\mathbb{D}_{KL}(\pi(y|x)||\pi^*(y|x))]=>\pi(y|x)=\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)exp(\frac{1}{\beta}r(x,y))

\end{aligned}
$$
最后一步：要让KL散度值最小，也就是两个分布一样，即得以上内容
$$
\begin{aligned}
&\pi(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)exp(\frac{1}{\beta}r(x,y))\\
&=>exp(\frac{1}{\beta}r(x,y))=\frac{\pi(y|x)}{\pi_{ref}(y|x)}Z(x)\\
&=>r(x,y)=\beta ln(\frac{\pi(y|x)}{\pi_{ref}(y|x)}Z(x))\\
&=>r(x,y)=\beta ln(\frac{\pi(y|x)}{\pi_{ref}(y|x)}Z(x))+\beta lnZ(x)
\end{aligned}
$$
得到**reward function**的一个表示

又知道 **Bradley Terry**中的Loss函数为：
$$
-ln\sigma(r(x,y_w)-r(x,y_l))
$$
代入r的表达式，可得：
$$
-\ln\sigma(\beta ln\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)}Z(x)+\beta lnZ(x)-\beta ln\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}-\beta lnZ(x))
$$
**DPO Loss:**
$$
-\ln\sigma(\beta ln\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)}Z(x)-\beta ln\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)})
$$
$\sigma(x)=\frac{1}{1+exp(-x)}$

