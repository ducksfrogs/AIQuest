w <- women
plot(w)
w_lm <- lm(weight ~ height, data = w)
summary(w_lm)
