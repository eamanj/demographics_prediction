library(ggplot2)
library(grid)

setwd(dir = "~/research/age_prediction/")
input_filename = "./data/european/cleaned_european_data_a2_c2_allweek_allday.csv.10%sample"
data = read.csv(file = input_filename, header = TRUE, sep = ",")
names(data)[names(data)=="attributes__survey_age"] = "age"
names(data)[names(data)=="attributes__survey_gender"] = "gender"
max_age = max(data$age)
min_age = min(data$age)
max_age = 90
min_age = 20
data = data[data$age >= min_age,]
data = data[data$age <= max_age,]
data$normalized_age = (data$age - min_age) / (max_age - min_age)


train_size = floor(nrow(data) * 0.75)
data = data[sample(nrow(data)), ]
train_data = data[1:train_size,]
train_data = train_data[!is.na(train_data$age),]
test_data = data[(train_size+1):nrow(data),]
test_data = test_data[!is.na(test_data$age),]
model = glm(normalized_age ~ . - gender - age, data = train_data, family = "quasibinomial")

train_data$normalized_age_predicted = predict(model, newdata = train_data, type = "response")
train_data$age_predicted = train_data$normalized_age_predicted * (max_age - min_age) + min_age
train_data = train_data[!is.na(train_data$age_predicted),]
quantile(train_data$age, c(0.05, 0.1, 0.9, 0.95, 0.99))
quantile(train_data$age_predicted, c(0.05, 0.1, 0.9, 0.95, 0.99))

test_data$normalized_age_predicted = predict(model, newdata = test_data, type = "response")
test_data$age_predicted = test_data$normalized_age_predicted * (max_age - min_age) + min_age
test_data = test_data[!is.na(test_data$age_predicted),]
quantile(test_data$age, c(0.05, 0.1, 0.9, 0.95, 0.99))
quantile(test_data$age_predicted, c(0.05, 0.1, 0.9, 0.95, 0.99))

# calibrate predictions
high_q = 0.95
low_q = 0.05
m = ((quantile(train_data$age, high_q) - quantile(train_data$age, low_q)) /
     (quantile(train_data$age_predicted, high_q) - quantile(train_data$age_predicted, low_q)))
b = quantile(train_data$age, high_q) - (m * quantile(train_data$age_predicted, high_q))
train_data$age_predicted_calibrated = m * train_data$age_predicted + b
test_data$age_predicted_calibrated = m * test_data$age_predicted + b

x_var = "actual"
y_var = "predicted"
xlabel = "Actual Age"
ylabel = "Predicted Age"
legend_title = "Age Groups"
bin_width = 10
rng = seq(min_age, max_age, bin_width)
#tick_labels = seq(min_rng + bin_width/2, max_rng, 2*bin_width)
#tick_labels =  ggplot2:::interleave(seq(min_rng + bin_width/2, max_rng,by=2*bin_width), "")
#tick_labels =  seq(min_rng + bin_width/2, max_rng,by=bin_width)
legend_labels = paste(seq(min_age, max_age-bin_width, bin_width),
                      seq(min_age+bin_width, max_age, bin_width), sep="-")
tick_labels = legend_labels
tmp_df = data.frame(x = cut(test_data$age, breaks=rng), y = test_data$age_predicted)
tmp_df = data.frame(x = cut(test_data$age, breaks=rng), y = test_data$age_predicted_calibrated)

output_filename = "./plots/eu_age_boxplot1.pdf"
ggplot(data = tmp_df, aes(x=x, y=y)) +
  theme_bw() +
  theme(panel.border = element_rect(fill=NA, colour = "black", size=1),
        legend.position="none",
        plot.margin=unit(c(0.3,0,0.4,0.2),"cm"),
        #legend.key = element_blank(),
        axis.text = element_text(size=14),
        axis.title.x = element_text(size=14,vjust=-0.6),
        axis.title.y = element_text(size=14)) +
        #legend.text = element_text(size=12),
        #legend.title = element_text(size=12)) +
        labs(x = xlabel, y = ylabel) +
        #guides(fill = guide_legend(title = legend_title)) +
        geom_boxplot(aes(fill = x)) +
        scale_x_discrete(labels=tick_labels) +
        #scale_fill_discrete(labels = legend_labels) +
        ylim(min_age, max_age)

ggsave(output_filename, width=6, height=4.5)
