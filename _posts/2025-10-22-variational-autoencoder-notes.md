---
layout: post
title: "VAE 简说：从隐变量到重参数化"
date: 2025-10-22 19:30:00 +0800
description: "从生成模型、变分后验、KL 散度、重构损失与重参数化技巧出发理解变分自编码器。"
tags: [vae, generative-model, deep-learning, fundamentals]
math: true
---

> 本文是关于变分自编码器（Variational Autoencoder, VAE）的个人学习笔记。

## 生成模型
生成模型的目标就是希望构建一个从隐变量z生成目标数据x的模型。更准确的讲，它们是假设了一个z服从某些常见的分布（正态分布或均匀分布），然后希望训练一个模型$X=g(Z)$，这个模型能讲原来的概率分布映射到训练的概率分布上去，也就是说他们的目的是进行分布之间的变换。
![从隐变量分布生成目标数据]({{ '/assets/images/blog/vae/generative-model.png' | relative_url }})
那现在假设Z服从标准的正态分布，那么我就可以从中采样得到若干个$Z_1,Z_2,…,Z_n$，然后对它做变换得到$X^1=g(Z_1),X^2=g(Z_2),…,X^n=g(Z_n)$，我们怎么判断这个通过g构造出来的数据集，它的分布跟我们目标的数据集分布是不是一样的呢？可以用KL散度吗？当然不行，因为KL散度是根据两个概率分布的表达式来算它们的相似度的，然而目前我们并不知道它们的概率分布的表达式，我们只有一批从构造的分布采样而来的数据${X^1,X^2,…,X^n}$，还有一批从真实的分布采样而来的数据${X1,X2,…,Xn}$（也就是我们希望生成的训练集）。我们只有样本本身，没有分布表达式，当然也就没有方法算KL散度。GAN的思路比较粗犷，没有合适的度量那就把这个度量使用神经网络训练出来。
## 什么是VAE
How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets?
directed probabilistic models: 有向概率模型，指的就是贝叶斯网络、即数据x是由某个隐变量z生成的，即（z -> x）
continuous latent variables with intractable posterior distributions: 连续的隐变量，并且其后验分布p(z|x)是难以计算的，即当我们观察到数据x后，想要反推它是由哪个z生成的，无法写出一个简单的表达式。
也就是说：
我们有一个很好的生成模型，但无法高效地用它进行“推断”（根据x找z），也无法大数据来学习模型的参数。
![VAE 的编码器与解码器结构]({{ '/assets/images/blog/vae/vae-overview.jpg' | relative_url }})
首先，我们有一批数据样本${X_1,X_2,...,X_n}$,整体使用$X$描述，我们想要通过$\{X_1, ..., X_n\}$得到$X$的分布$p(X)$,如果能得到的话，我直接根据$p(X)$采样，就可以得到所有可能的$X$了（包括样本之外的X）,这是一个终极理想的生成模型。当然，这个很难实现。于是我们希望能学到一个真实的分布$p(x)$

$$
p(X) = \sum_Z p(X|Z) p(Z)
$$

其中$p(x|z)$是Likelihood条件分布，$p(z)$是先验分布（prior distribution）
这里，我们不分求和还是积分，意思对了就行，此时$p(X|Z)$描述了一个由Z生成X的模型，我们假设Z服从标准正态分布，即$p(Z) = \mathcal{N}(0, I)$，如果这个理想能实现，我们可以从一个标准正态分布中采样一个Z，然后根据Z来计算一个X，也是一个不错的生成模型。如果以这种方式，就必须要穷尽所有z。
![VAE 的传统理解示意图]({{ '/assets/images/blog/vae/latent-variable-model.png' | relative_url }})
上图是对VAE的传统理解，我们其实不清楚，究竟经过重新采样出来的$Z_k$是不是还对应着原来的$X_k$,所以我没如果最小化$\mathbb D{(\hat X_k, X_k)}^2$ (这是个距离函数)是不科学的，事实上，如果你查看官方给出的代码也不是这样实现的。
其实在整个VAE模型中，我们并没有去使用$p(Z)$是正态分布的假设，而我们假设后验分布$p(Z|X)$是正态分布。也就是使用它去近似后验分布。如何把先验分布与后验分布联系在一起呢？

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}
$$

其实p(z|x)就是Encoder, p(x|z)就是Decoder。
给定一个真实样本$X_k$，我们假设存在一个专属于它的分布$p(Z|X_k)$，这就是后验分布，并假设这个分布是独立的，多元的正态分布。为什么要强调专属呢？因为我们后面要训练一个生成器$X=g(Z)$，希望能从分布$p(Z|X_k)$采样出来的$Z_k$还原成$X_k$,如果假设$p(Z)$是正态分布，然后从$p(Z)$中采样一个Z，我们无法知道这个Z对应哪个真实的$X$。现在分布$p(Z|X_k)$专属于$X_k$，我们有理由说从这个分布中采样的Z应该要还原到$X_k$中去。
![样本对应的近似后验分布]({{ '/assets/images/blog/vae/posterior-distribution.png' | relative_url }})
这个就是VAE的理解上的示意图。
VAE让所有的$p(Z|X)$都向标准正态看齐，于是对于Z的采样：

$$
p(Z) = \sum_{X} p(Z|X)p(X) = \sum_{X} \mathcal{N}(0, I)p(X) = \mathcal{N}(0, I) \sum_{X} p(X) = \mathcal{N}(0, I)
$$

于是我们的先验假设$P(z)$是标准正态分布，我们就可以放心从标准正态分布中采样用于生成X了。
那么如何让所有的$p(Z|X)$都向标准正态分布看齐呢？最直接的办法就是让$p(Z|X)$向标准正态分布看齐，于是目标函数我们也有了：

$$
\mathcal{L}_{\mu, \sigma^2} =  D_{KL}(q(z|x)||p(z|x)) = E_{z \sim q(z|x)}[\log q(z|x) - \log p(z|x)]
$$

这里为了严谨点，也向普遍的VAE教程和论文看齐，$p(Z|X)$为我们假设的标准正态分布$\mathcal{N}(0, I)$，$q(Z|X)$是我们训练中产生的正态分布$\mathcal{N}(\mu, \sigma^2)$
>NOTE:
>$D_{KL}(P || Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)$

于是，公式可以变为：

$$
\mathcal{L}_{\mu, \sigma^2} =  D_{KL}(\mathcal{N}(\mu, \sigma^2)||\mathcal{N}(0, I)) = E_{z \sim \mathcal{N}(\mu, \sigma^2)}[\log \mathcal{N}(\mu, \sigma^2) - \log \mathcal{N}(0, I)]
$$

进一步计算：

$$
\mathcal{L}_{\mu, \sigma^{2}} = \frac{1}{2} \sum_{i=1}^{d} \left( \mu_{(i)}^{2} + \sigma_{(i)}^{2} - \log \sigma_{(i)}^{2} - 1 \right)
$$

推导公式：

$$
\begin{aligned}
KL(N(\mu, \sigma^2)\|N(0,1))
&=\int \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2} \left( \log \frac{e^{-(x-\mu)^2/2\sigma^2}/\sqrt{2\pi\sigma^2}}{e^{-x^2/2}/\sqrt{2\pi}} \right) dx  \\
&=\int \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2} \log \left\{ \frac{1}{\sqrt{\sigma^2}} \exp \left\{ \frac{1}{2}[x^2-(x-\mu)^2/\sigma^2] \right\} \right\} dx \\
&=\frac{1}{2} \int \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2} \left[-\log \sigma^2 + x^2 - (x-\mu)^2/\sigma^2 \right] dx 
\end{aligned}
$$

这样结束了吗？实际上是没有的，这个解决了$P(Z|X)$的也就是我们所谓的编码器的均值方差模型优化，那么如何衡量由Z生成（重构）X的能力呢？那么我们可以直接计算重构数据与原始数据的差异，如果是MNIST，它数值是二值的，那么$p(x|z)$通常是一个伯努利分布，此时可以使用交叉熵损失，如果是一个高斯分布（像素归一化到 [0, 1]）此时可以使用最小均方误差损失。我们将其统一表示为最大似然估计，去最大似然它.那么我们的重构损失为：

$$
\mathcal l = - \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}\left[\log p_{\theta}(\mathbf{x}|\mathbf{z})\right]
$$

$$
\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \log p_{\theta}(\mathbf{x}|\mathbf{z}) \right] = \int q_{\phi}(\mathbf{z}|\mathbf{x}) \log p_{\theta}(\mathbf{x}|\mathbf{z}) d\mathbf{z}
$$

举例说明：

MNIST：

![MNIST 数据的重构损失]({{ '/assets/images/blog/vae/mnist-reconstruction.png' | relative_url }})

连续数据：

![连续数据的重构损失]({{ '/assets/images/blog/vae/continuous-reconstruction.png' | relative_url }})

为了简单高效，只需要采样1个或z个样本，然后用这些样本的均值实现，那么就有了：

![使用单个样本估计期望]({{ '/assets/images/blog/vae/monte-carlo-estimate.png' | relative_url }})

那么一个样本能准确吗？参数重整化和随机梯度下降，能保证所有数据点和所有随机采样的平均效果都会收敛到真实的期望值。

## 重参数技巧
我们要从$P(Z|X_k)$中采样一个$Z_k$出来，尽管我们知道$P(Z|X_k)$是一个正态分布，但是均值和方差都是靠模型算出来的，我们要靠这个过程来优化均值方差的模型，但是“采样这个操作”是不可导的，因为采样是随机的，他并不可微。但是采样的结果是可导的。

$$
\begin{align*}
&\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)dz \\
&=\frac{1}{\sqrt{2\pi}}\exp\left[-\frac{1}{2}\left(\frac{z-\mu}{\sigma}\right)^2\right]d\left(\frac{z-\mu}{\sigma}\right)
\end{align*}
$$

$(z - \mu)/\sigma = \varepsilon$ 它是一个标准正态分布，而$\varepsilon dx$是概率，于是从正态分布中采样一个Z，相当于从正态分布中采样一个$\varepsilon$再让$Z = \mu + \varepsilon \times \sigma$。这样一来，“采样”这个操作就不用参与梯度下降了，改为采样的结果参与，使得整个模型可训练了。

推荐阅读：苏剑林的[科学空间](https://spaces.ac.cn/)。
