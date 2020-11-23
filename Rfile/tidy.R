library(tidyverse)
library(nycflights13)
table1
table2
table3
table1 %>%
  mutate(rate=cases /population * 10000)
table1 %>% 
  count(year, wt=cases)
library(ggplot2)
ggplot(table1, aes(year, cases)) +
  geom_line(aes(group= country), colour='grey50') +
  geom_point(aes(colour= country))

table4a
table4a %>%
  pivot_longer(c('1999', '2000'), names_to = "year", values_to="cases")

table4b %>%
  pivot_longer(c('1999','2000'), names_to = 'year', values_to='population')

tidy4a <- table4a %>% pivot_longer(c('1999', '2000'),
                         names_to = "year", 
                         values_to="cases")

tidy4b <- table4b %>% pivot_longer(c('1999','2000'), names_to = 'year', values_to='population')
left_join(tidy4a, tidy4b)

table2
table2 %>%
  pivot_wider(names_from = type, values_from = count)


stocks <- tibble(
  year = c(2015, 2015, 2016, 2016),
  half =  c(1,2,1,2),
  return = c(1.88, 0.59, 0.92, 0.17)
)

stocks %>% 
  pivot_wider(names_from = year, values_from = return) %>%
  pivot_longer('2015':'2016', names_to='year', values_to="return")

table4a %>% 
  pivot_longer(c(1999, 2000), names_to = "year", values_to = "cases")

