x <- seq(-3,3,0.1)
y <- dt(x, df=19)
plot(x,y,type = 'l')
qt(0.025, df=19)

library(tidyverse)
irisdata <- iris
summary(irisdata)

x <- iris
data_mean_sd <- x %>%
  group_by(Species) %>%
  summarise(mean = mean(Sepal.Length), sd = sd(Sepal.Length))
data_mean_sd
errors <- aes(ymax=mean + sd, ymin= mean -sd)
errors
box <- ggplot(data = x,
              aes(x=Species, y=Sepal.Length, fill=Species)) +
  labs(x='', y='Sepal.Length') +
  scale_y_continuous(expand = c(0,0), limits = c(0,ylim)) +
  theme_classic()+
  scale_fill_manual(values = c('gray20','gray60', 'gray80')) +
  stat_boxplot()
box
boxplot(irisdata$Sepal.Length)
ggplot(data = irisdata, aes(x=Species, y=Sepal.Length))
