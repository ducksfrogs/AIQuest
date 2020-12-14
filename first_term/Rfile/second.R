library(tidyverse)

seq(1, 10)
seq(2,10)

y <- seq(1, 10, length.out = 5)
y
library(nycflights13)
flights
filter(flights, month==1, day==1)

dec25 <- filter(flights, month==12, day==25)
dec25
filter(flights, month==1)
filter(flights, month ==11 | month==12)

df <- tibble(x = c(1,NA,3))
df
filter(df, x >1)
filter(df, is.na(x) | x >1)

arrange(flights, year, month, day)
arrange(flights, desc(dep_delay))

df <- tibble(x=c(5,2,NA))
df
arrange(df, x)
arrange(df, desc(x))
filter(flights, month %in% c(11,12))

filter(flights, !(arr_delay >120 | dep_delay > 120))
filter(flights, arr_delay <= 120, dep_delay <= 120)

select(flights, year, month, day)
select(flights, -(year:day))
flights
select(flights, time_hour, air_time, everything())
