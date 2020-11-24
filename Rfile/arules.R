install.packages('arules')
library(arules)
data("Groceries")

g0 <- Groceries
gfrm0 <- as(g0, 'data.frame')
gdat1 <- itemFrequency(g0)
itemFrequencyPlot(g0)
Groceries
r1 <-apriori(g0, parameter = list(confidence=0.5, support=0.01))
r3 <- sort(r1, d=T, by='confidence')

inspect(r3)

r4 <-apriori(g0, parameter = list(confidence=0.5, support=0.009))
r5 <- sort(r4, d=T, by='confidence')

inspect(r5)

head(gfrm0)
summary(gfrm0)
summary(g0)
itemFrequency(g0)
sort(gdat1)
