library(tidyverse)
ggplot(data=mpg) +
  geom_point(mapping = aes(x=displ, y=hwy, color=class))

ggplot(data = mpg) +
  geom_point(mapping = aes(x=displ, y=hwy)) +
  facet_wrap(~class)


ggplot(data = mpg) +
  geom_point(mapping = aes(x=displ, y=hwy)) +
  facet_wrap(~class, nrow = 2)

ggplot(data = mpg) +
  geom_point(mapping = aes(x=displ, y=hwy)) +
  facet_grid(drv~cyl)
head(mpg)
table(mpg$cyl)

ggplot(data = diamonds) +
  geom_bar(mapping = aes(x= cut))

ggplot(data = diamonds) +
  stat_count(mapping = aes(x= cut))


demo <- tribble(
  ~cut, ~freq,
  "Fair", 1610,
  "Good", 4986,
  "Very Good", 12082,
  "Premium", 13791,
  "Ideal", 21551
)
ggplot(data = demo) +
  geom_bar(mapping = aes(x=cut, y=freq), stat = 'identity')

ggplot(data = mpg, mapping = aes(x=class, y=hwy)) +
  geom_boxplot()

ggplot(data = mpg, mapping = aes(x=class, y=hwy)) +
  geom_boxplot() +
  coord_flip()
bar <- ggplot(data = diamonds)+
  geom_bar(
    mapping = aes(x=cut, fill=cut),
    show.legend = FALSE,
    width = 1
  ) +
  theme(aspect.ratio = 1) +
  labs(x=NULL, y=NULL)

bar + coord_flip()
bar + coord_polar()
