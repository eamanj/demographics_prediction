library(ggplot2)
library(grid)

age_boxplot = function(full_input_file, full_output_file, y_min, y_max, width, height) {
  x_var = "actual"
  y_var = "predicted"
  xlabel = "Actual Age"
  ylabel = "Predicted Age"
  legend_title = "Age Groups"
  data = read.csv(file = full_input_file, header = TRUE, sep = ",")
  
  min_rng= 20
  max_rng=90
  data = data[data$predicted>y_min,]
  data = data[data$predicted<y_max,]
  data = data[data$actual>min_rng,]
  data = data[data$actual<=max_rng,]
  predicted = data$predicted
  actual = data$actual
  bin_width = 10
  rng = seq(min_rng, max_rng, bin_width)
  #tick_labels = seq(min_rng + bin_width/2, max_rng, 2*bin_width)
  #tick_labels =  ggplot2:::interleave(seq(min_rng + bin_width/2, max_rng,by=2*bin_width), "")
  #tick_labels =  seq(min_rng + bin_width/2, max_rng,by=bin_width)
  legend_labels = paste(seq(min_rng, max_rng-bin_width, bin_width),
                        seq(min_rng+bin_width, max_rng, bin_width), sep="-")
  x_tick_labels = legend_labels
  y_tick_labels = seq(0, 100, 2*bin_width)
  tmp_df = data.frame(x = cut(actual, breaks=rng), y = predicted)
  
  pdf(full_output_file, width=width, height=height)
  print(ggplot(data = tmp_df, aes(x=x, y=y)) +
    theme_bw() +
    theme(panel.border = element_rect(fill=NA, colour = "black", size=1),
          legend.position="none",
          plot.margin=unit(c(0.3,0.5,0.4,0.2),"cm"),
          axis.text = element_text(size=14),
          axis.title.x = element_text(size=14,vjust=-0.6),
          axis.title.y = element_text(size=14)) +
          labs(x = xlabel, y = ylabel) +
          geom_boxplot(fill = "dodgerblue2") +
          scale_x_discrete(labels=x_tick_labels) +
          scale_y_continuous(breaks=y_tick_labels, limits=c(y_min, y_max)))
  dev.off()
}


setwd(dir = "~/research/age_prediction/")
input_files = dir("./results/elastic-net/european/test_predictions/")
for(input_file in input_files) {
  output_file = gsub("%", "percent", input_file, fixed=TRUE)
  full_output_file = paste0("./plots/elastic-net/european/", output_file, ".pdf")
  full_input_file = paste0("./results/elastic-net/european/test_predictions/", input_file)
  age_boxplot(full_input_file, full_output_file, y_min=0, y_max=100, width=6, height=4.5)
}

input_files = dir("./results/linear/european/test_predictions/")
for(input_file in input_files) {
  output_file = gsub("%", "percent", input_file, fixed=TRUE)
  full_output_file = paste0("./plots/linear/european/", output_file, ".pdf")
  full_input_file = paste0("./results/linear/european/test_predictions/", input_file)
  age_boxplot(full_input_file, full_output_file, y_min=0, y_max=100, width=6, height=4.5)
}