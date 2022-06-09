#before start do (RConsole) : options(expressions=500000), 
# options(max.print=1000000)
############################################################
library(keras)
library(tensorflow)
library(EBImage)

setwd('D://Tugas Akhir')
save_in <- ("D://Tugas Akhir Run1000/")

images <- list.files()
w <- 100
h <- 100
for(i in 1:length(images))
{
  result <- tryCatch({
    # Image name
    imgname <- images[i]
    # Read image
    img <- readImage(imgname)
    # Path to file
    img_resized <- resize(img, w = w, h = h)
    path <- paste(save_in, imgname, sep = "")
    # Save image
    writeImage(img_resized, path, quality = 70)
    # Print status
    print(paste("Done",i,sep = " "))},
    # Error function
    error = function(e){print(e)})
}

setwd("D://Tugas Akhir Run1000")
# Read Images
images <- list.files()
images
summary(images)
list_of_images = lapply(images, readImage)
head(list_of_images)
display(list_of_images[[15]])
tail(list_of_images)


#create train D2
train <- list_of_images[c(1:40, 121:160, 161:200, 
                          201:240, 321:360, 361:400,
                          401:440, 521:560, 561:600,
                          601:640, 721:760, 761:800,
                          801:840, 921:960, 961:1000)]
str(train)
display(train[[20]])
display(train[[110]])

#create test
test <- list_of_images[c(41:80, 241:280, 441:480, 641:680, 841:880)]
test
display(test[[1]])


par(mfrow = c(10,10))
for (i in 1:600) plot(train[[i]])

# Resize & combine
str(train)
for (i in 1:600) {train[[i]] <- resize(train[[i]], 32, 32)}
for (i in 1:200) {test[[i]] <- resize(test[[i]], 32, 32)}
# for (f in 1:800) {print(dim(train[[f]]))}

train <- combine(train)
str(train)
x <- tile(train, 20)
display(x, title='train')

test <- combine(test)
str(test)
y <- tile(test, 10)
display(y, title = 'test')


# Reorder dimension
train <- aperm(train, c(4,1,2,3))
test <- aperm(test, c(4,1,2,3))
str(train)
str(test)

# Response/Give Label for the Data
trainy <- c(rep(0,40), rep(0,40), rep(0,40),
            rep(1,40), rep(1,40), rep(1,40),
            rep(2,40), rep(2,40), rep(2,40),
            rep(3,40), rep(3,40), rep(3,40),
            rep(4,40), rep(4,40), rep(4,40))
testy <- c(rep(0,40), rep(1,40), rep(2,40), rep(3,40), rep(4,40))

# One hot encoding
trainLabels <- to_categorical(trainy)
trainLabels
testLabels <- to_categorical(testy)
testLabels

# Model
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = c(32, 32, 3)) %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.01) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.01) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate=0.01) %>%
  layer_dense(units = 5, activation = 'softmax') %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_sgd(lr = 0.01,
                                    decay = 1e-6,
                                    momentum = 0.9,
                                    nesterov = T),
          metrics = c('accuracy'))
summary(model)

# Fit model
history <- model %>%
  fit(train,
      trainLabels,
      epochs = 50, 
      batch_size = 32,
      validation_split = 0.1,
      validation_data = list(test, testLabels))
plot(history)


# Evaluation & Prediction - train data
model %>% evaluate(train, trainLabels)
pred <- model %>% predict_classes(train)
table(Predicted = pred, Actual = trainy)
prob <- model %>% predict_proba(train)
colnames(prob)<- c('Belum Matang','Bagus Grid 1','Bagus Grid 2','Busuk','Rusak')
cbind(prob, Predicted_class = pred, Actual = trainy)


# Evaluation & Prediction - test data
model %>% evaluate(test, testLabels)
pred <- model %>% predict_classes(test)
table(Predicted = pred, Actual = testy)
prob <- model %>% predict_proba(test)
colnames(prob)<- c('Belum Matang','Bagus Grid 1','Bagus Grid 2','Busuk','Rusak')
cbind(prob, Predicted_class = pred, Actual = testy)


#save model
save_model_weights_hdf5(model,filepath='D://TugasAkhirRunD2.hdf',overwrite=TRUE)


#Testing
model=load_model_weights_hdf5(model,filepath="D://TugasAkhirRunD2.hdf",by_name=FALSE)
setwd('D:/Tugas Akhir Run1000')

images <- list.files()
images
summary(images)
list_of_images = lapply(images, readImage)
head(list_of_images)
display(list_of_images[[3]])


#create test
testt <- list_of_images[c(81:120, 281:320, 481:520, 681:720, 881:920)]
testt
display(testt[[1]])

for (i in 1:200) {testt[[i]] <- resize(testt[[i]], 32, 32)}
testing <- combine(testt)
str(testing)
y <- tile(testing, 10)
display(y, title = 'test')
str(testing)

Uji <- aperm(testing, c(4, 1, 2, 3))
str(Uji)
testy <- c(rep(0,40), rep(1,40), rep(2,40), rep(3,40), rep(4,40))
testLabels <- to_categorical(testy)

pred <- model %>% predict_classes(Uji)
model %>% evaluate(Uji, testLabels)
table(Predicted = pred, Actual = testy)
prob <- model %>% predict_proba(Uji)
prob
colnames(prob)<- c('Belum Matang','Bagus Grid 1','Bagus Grid 2','Busuk','Rusak')
cbind(prob, Predicted_class = pred, Actual = testy)

