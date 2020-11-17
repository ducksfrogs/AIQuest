car_data <- cars
res <- glm(speed ~ dist, data = car_data)
summary(res)
res$coefficients
coef(res)
plot(car_data$speed, car_data$dist)
abline(res)
res2 <- glm(dist ~ speed, data = car_data)

plot(car_data$speed, car_data$dist)
abline(res2)
