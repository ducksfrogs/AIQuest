library(rpart)
s1 <- read.csv("exam.csv", header = T, row.names = 1)
s2 <- rpart(score~age + sex + trial, data = s1, method = 'class')
library(rpart.plot)
rpart.plot(s2, extra = 2)
table(s1)
summary(s1)
s2
