# Monlan
Monlan is a collection of my Data Science (including Reinforcement Learning) experiments (2019-2022) in the field of algorithmic trading. 
The main idea was to build composite agent which
consists of independent special agents: special opener agent, which analyzes situation for
opening buy or sell deal or do nothing (hold position, wait better situation for opening), and special
buy and sell agents, which know better, when to hold or close position. When opener choose
buy or sell action, management is going to special buy or sell agent. They all are independent.


The last version, which you can run through scripts directly, is the latest most advanced and clean version 
(much more effective design of enviroment, agents; high performance learning and rendering; best results; bug fixes).
Previous versions in the corresponding folder contain a lot of intresting ideas and realisations, 
which may be useful for you (integrations with trading terminal, risk management agent, price level
analyzis, collections of feature generators, different architectures and approaches).

# The main idea of the current version (Big Supervised)

The main idea of Big Supervised Monlan is generating stochastic 
value estimation for each history step. Then there's training a model that 
predicts values for buy and sell. Based on this model we're building 
mirror closer: if it plays the role of a buyer, then its hold action value estimates 
as buyer value prediction and close action value estimates as sell value prediction. 
(A mirror closer in the role of buyer holds position while it is more profitable
than close and switch to the seller.) 
Then a trained mirror closer is using for generating real performance estimation 
for an opener in each history point of a training data. Then we add 
hold estimates to these targets and train an opener. It is important because 
no one can trade profitable without the ability to wait a good moment to open a deal. 
After that we combine the trained opener and the mirror closer into one 
CompositeAgent that is testing on a test data in the last script. 
This idea came from experiments with pure RL and distilled agents. 
This method combined with on-fly training sample generation makes 
it possible to perform relatively fast and effective training of an agent on a big data (millions of history samples) 
on a modern laptop (64 Gb DDR4 RAM, i7-11800H, RTX3060 6Gb). To complete the whole pipeline from scratch one needs ~5-7 days. 

# Example of results on train data subsample

(download the image below to see details better)

![](resnet_18_naive_opener_0_16_train_subsample_73000.png)

# Example of results on test data subsample

(download the image below to see details better)

![](resnet_18_naive_opener_0_16_test_subsample_73000.png)

# Installation

```
git clone https://github.com/CameleoGrey/Monlan.git
conda create -n monlan python=3.9
conda activate monlan
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```

# Note
Remember: when starting to work in the financial markets, make sure that 
you are aware of the risks associated with trading with leverage, and that 
you have a sufficient level of training. You shall assume the risk of financial loss 
caused by actions of robots that you've created by this library.

