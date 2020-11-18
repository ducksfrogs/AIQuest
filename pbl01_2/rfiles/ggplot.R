library(tidyverse)
library(lme4)
library(MASS)
library(ggeffects)
install.packages('ggeffects')
install.packages(c('nycflights13', 'gapminder','Lahman'))
mpg
ggplot(data = mpg) +
  geom_point(mapping = aes(x= displ, y=hwy, color=class))
# Left
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, alpha = class))

# Right
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, shape = class))
ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy), color = "blue")
ggplot(data = mpg) 
+ geom_point(mapping = aes(x = displ, y = hwy))

ggplot(data = irisdata) +
  geom_point(mapping = aes(x=Sepal.Length, y=Petal.Length, color=Species))
data(father.son)
install.packages("UsingR")
library(UsingR)
data("father.son")
fs <- father.son
cor(fs$fheight, fs$sheight)
cor(women$height, women$weight)
ggplot(data = women) +
  geom_point(aes(x=weight, y=height))
w_glm <- glm(weight ~ height, data = women)
w_lm
w_glm
wglm_p <- ggpredict(w_glm, terms = 'height')
plot(wglm_p)
wglm_p


ggplot() +
  geom_point(data= women, aes(x=height, y=weight)) +
  geom_line(data=wglm_p, aes(x=x, y=predicted)) +
  geom_ribbon(data = wglm_p, aes(x=x, ymin=conf.low, ymax=conf.high), alpha=0.2)
