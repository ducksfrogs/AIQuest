a <- 1:5
a
tibble(a, a *2)
tibble(a, b=a*2, c=1)
tibble(x = runif(10), y = x*2)
runif(5)
df <- tibble(`a 1` =1, `a 2` = 2)
df[1]
