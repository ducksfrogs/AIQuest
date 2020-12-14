require(stats)
require(graphics)
plot(cars, xlab="Speed(mph)", ylab="STopping distance (ft)",
     las=1)
library(tidyverse)
?ggplot2
?cars
lines(lowess(cars$speed, cars$dist, f=2/3, iter = 3), col='red')
?lowess
plot(cars, main="lowess(cars)")
plot(cars, xlab="Speed(mph)", ylab="STopping distance (ft)",
     las=1, log='xy')
title(main = "cars data (logarithmic scales")
summary(fm1 <- lm(log(dist) ~ log(speed), data = cars))
plot(fm1)

