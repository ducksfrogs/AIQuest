x <- seq(-3,3,0.1)
y <- dt(x, df=19)
plot(x,y,type = 'l')
qt(0.025, df=19)
irisdata <- iris