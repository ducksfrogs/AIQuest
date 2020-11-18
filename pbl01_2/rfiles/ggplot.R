library(tidyverse)
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
