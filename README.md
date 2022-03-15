# Monlan
Monlan is a collection of my Reinforcement Learning experiments (2019-2022) 
in the field of algorithmic trading. The main idea was to build composite agent which
consists of independent special agents: special opener agent, which analyzes situation for
opening buy or sell deal or do nothing (hold position, wait better situation for opening), and special
buy and sell agents, which know better, when to hold or close position. When opener choose
buy or sell action, management is going to special buy or sell agent. They all are independent.


The last version, which you can run through scripts directly, is the latest most advanced and clean version 
(much more effective design of enviroment, agents; high performance learning and rendering; best results; bug fixes).
Previous versions in the corresponding folder contain a lot of intresting ideas and realisations, 
which may be useful for you (integrations with trading terminal, risk management agent, price level
analyzis, collections of feature generators, different architectures and approaches).

# Example of render

![](example_test_plot.png)

# Installation

```
git clone https://github.com/CameleoGrey/Monlan.git
conda create -n monlan python=3.9
conda activate monalan
pip install -r requirements.txt
```

# Some thoughts

I started developing Monlan in May 2019.
After that, I continued my research until February 2022 and created many versions of Monlan,
but did not achieve positive results on test ("future") data. I really tried to build a profitable robot. 
There was everything: thousands of experiments, 3-4 depressions, loss of money. While I was working on robots, I learned how to trade without 
them, with my hands (it turned out I was doing pretty well, but it turned out to be too boring for me, so I 
continued to build robots). Please, if you are just starting with
algorithmic trading, do not continue, do not waste your precious time. If you want to earn money, it's much 
easier to do it at a regular job in a company. Trading is a pseudoscientific field where works: 
arbitrage (prohibited by every broker), short-term scalping (available only for large
companies), medium-term hand trading based on macroeconomic analysis, where technical
analysis plays only a supporting role (requires a lot of time for training, talent and does not guarantee huge profits).
Spending time creating trading bots was one of my biggest failures because I spent 3 years on
an unsolvable task. If you want to understand the difference between scientific and
pseudoscientific tasks, look for the concepts of "holism" and "atomism" in the methodology of science. 
Real science is based on atomism, not holism. Forecasting the future price based on the price in the past in isolation from understanding the economic situation is holism.

# Note
Remember: when starting to work in the financial markets, make sure that 
you are aware of the risks associated with trading with leverage, and that 
you have a sufficient level of training. You shall assume the risk of financial loss 
caused by actions of robots that you've created by this library.

