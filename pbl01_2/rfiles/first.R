data("iris")
str(iris)
summary(iris)
library(ggplot2)
ggplot(iris, aes(x=Sepal.Length, y=Petal.Width)) 
+ geom_point(color=Species)
diamonds
iris
library(tidyverse)
iris_tbl <- as_tibble(iris)
ggplot(iris_tbl, aes(x=Sepal.Length, y=Petal.Width)) +
  geom_point(aes(color= Species))
data(father.son)
install.packages('UsingR')
data((father.son))
library(UsingR)
father.son
plot(father.son)
fs <- father.son
glm(fs$fheight ~ fs$sheight, data = fs)
lm(fs$fheight ~ fs$sheight, data = fs)
