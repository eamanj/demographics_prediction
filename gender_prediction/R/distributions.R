################################################
setwd(dir = "~/research/gender_prediction/data/south_asian/")
south_asian_data = read.csv(file = "cleaned_south_asian_data_a2_c2.csv", header = TRUE, sep = ",")
south_asian_male_data = south_asian_data[south_asian_data$attributes__surv_gender == 1,]
south_asian_female_data = south_asian_data[south_asian_data$attributes__surv_gender == 0,]

setwd(dir = "~/research/gender_prediction/data/european/")
european_data = read.csv(file = "cleaned_european_data_a2_c2.csv.30%sample", header = TRUE, sep = ",")
european_male_data = european_data[european_data$attributes__survey_gender == 1,]
european_female_data = european_data[european_data$attributes__survey_gender== 0,]

setwd(dir = "~/research/gender_prediction/data/central_american/")
central_american_data = read.csv(file = "cleaned_central_american_data_a2_c2.csv.30%sample", header = TRUE, sep = ",")
central_american_male_data = central_american_data[central_american_data$attributes__gender == 1,]
central_american_female_data = central_american_data[central_american_data$attributes__gender== 0,]

num_south_asian_data_points = nrow(south_asian_data)
num_european_data_points = nrow(european_data)
num_central_american_data_points = nrow(central_american_data)

# set the dir for output plots
setwd(dir = "~/research/gender_prediction/plots/distributions/")

generate_column_dist_plots = function(column,
                                      density_num_points,
                                      south_asian_num_bins,
                                      european_num_bins,
                                      central_american_num_bins,
                                      density_min_x,
                                      density_max_x,
                                      density_min_y,
                                      density_max_y,
                                      south_asian_histogram_min_x,
                                      south_asian_histogram_max_x,
                                      european_histogram_min_x,
                                      european_histogram_max_x,
                                      central_american_histogram_min_x,
                                      central_american_histogram_max_x,
                                      south_asian_density_min_x,
                                      south_asian_density_max_x,
                                      south_asian_density_min_y,
                                      south_asian_density_max_y,
                                      european_density_min_x,
                                      european_density_max_x,
                                      european_density_min_y,
                                      european_density_max_y,
                                      central_american_density_min_x,
                                      central_american_density_max_x,
                                      central_american_density_min_y,
                                      central_american_density_max_y,
                                      legend_position) {
density_adjust = 1
filename=paste0(column, ".png")
png(filename, width=1900, height=2100, units="px", pointsize = 25)
layout(matrix(c(1,2,5,1,3,6,1,4,7), 3, 3, byrow = TRUE))
options(scipen=3) 

south_asian_column_data = south_asian_data[[column]]
south_asian_male_column_data = south_asian_male_data[[column]]
south_asian_female_column_data = south_asian_female_data[[column]]

european_column_data = european_data[[column]]
european_male_column_data = european_male_data[[column]]
european_female_column_data = european_female_data[[column]]

central_american_column_data = central_american_data[[column]]
central_american_male_column_data = central_american_male_data[[column]]
central_american_female_column_data = central_american_female_data[[column]]

title = paste("Density of", column)
plot(density(european_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust),
     main=title,
     xlab=column, ylab="Density", col="blue", lwd = 1.5,
     xlim = c(density_min_x, density_max_x),
     ylim= c(density_min_y, density_max_y),
     cex.main=0.85, bty="n")
lines(density(south_asian_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust),
      col="indianred", lwd = 1.5)
lines(density(central_american_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust),
      col="forestgreen", lwd = 1.5)
legend(x = legend_position, legend=c("European", "South Asian", "Central American"),
       col=c("blue", "indianred", "forestgreen"), lty = 1, lwd = 3, box.lwd=0)



extra_info = paste("num_data_points =", length(south_asian_column_data), "\n",
                   "max =", signif(max(south_asian_column_data, na.rm = TRUE), 4),
                   "min =", signif(min(south_asian_column_data, na.rm = TRUE), 4), 
                   "median =", signif(median(south_asian_column_data, na.rm = TRUE), 4),
                   "mean =", signif(mean(south_asian_column_data, na.rm = TRUE), 4),
                   "std =", signif(sd(south_asian_column_data, na.rm = TRUE), 4))
title = paste("South asian histogram of", column)
hist(south_asian_column_data, main=title, xlab=column, ylab="Frequency",
     xlim = c(south_asian_histogram_min_x, south_asian_histogram_max_x),
     breaks=south_asian_num_bins, cex.main=0.85)
mtext(extra_info, side = 3, line = -0.15, cex=0.65)

extra_info = paste("num_data_points =", length(european_column_data), "\n",
                   "max =", signif(max(european_column_data, na.rm = TRUE), 4),
                   "min =", signif(min(european_column_data, na.rm = TRUE), 4), 
                   "median =", signif(median(european_column_data, na.rm = TRUE), 4),
                   "mean =", signif(mean(european_column_data, na.rm = TRUE), 4),
                   "std =", signif(sd(european_column_data, na.rm = TRUE), 4))
title = paste("European histogram of", column)
hist(european_column_data, main=title, xlab=column, ylab="Frequency",
     xlim = c(european_histogram_min_x, european_histogram_max_x),
     breaks=european_num_bins, cex.main=0.85)
mtext(extra_info, side = 3, line = -0.15, cex=0.65)

extra_info = paste("num_data_points =", length(central_american_column_data), "\n",
                   "max =", signif(max(central_american_column_data, na.rm = TRUE), 4),
                   "min =", signif(min(central_american_column_data, na.rm = TRUE), 4), 
                   "median =", signif(median(central_american_column_data, na.rm = TRUE), 4),
                   "mean =", signif(mean(central_american_column_data, na.rm = TRUE), 4),
                   "std =", signif(sd(central_american_column_data, na.rm = TRUE), 4))
title = paste("Central american histogram of", column)
hist(central_american_column_data, main=title, xlab=column, ylab="Frequency",
     xlim = c(central_american_histogram_min_x, central_american_histogram_max_x),
     breaks=central_american_num_bins, cex.main=0.85)
mtext(extra_info, side = 3, line = -0.15, cex=0.65)



extra_info = paste("Male: num_points =", length(south_asian_male_column_data),
                   "max =", signif(max(south_asian_male_column_data, na.rm = TRUE), 4),
                   "min =", signif(min(south_asian_male_column_data, na.rm = TRUE), 4), 
                   "median =", signif(median(south_asian_male_column_data, na.rm = TRUE), 4),
                   "mean =", signif(mean(south_asian_male_column_data, na.rm = TRUE), 4),
                   "std =", signif(sd(south_asian_male_column_data, na.rm = TRUE), 4), "\n",
                   "Female: num_points=", length(south_asian_female_column_data),
                   "max =", signif(max(south_asian_female_column_data, na.rm = TRUE), 4),
                   "min =", signif(min(south_asian_female_column_data, na.rm = TRUE), 4), 
                   "median =", signif(median(south_asian_female_column_data, na.rm = TRUE), 4),
                   "mean =", signif(mean(south_asian_female_column_data, na.rm = TRUE), 4),
                   "std =", signif(sd(south_asian_female_column_data, na.rm = TRUE), 4))
title = paste("South asian density of", column, "indicator")
plot(density(south_asian_male_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust),
     main=title,
     xlim = c(south_asian_density_min_x, south_asian_density_max_x),
     ylim= c(south_asian_density_min_y, south_asian_density_max_y),
     xlab=column, ylab="Density", col="blue", lwd = 1.5, cex.main=0.85, bty="n")
lines(density(south_asian_female_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust),
      col="indianred", lwd = 1.5)
mtext(extra_info, side = 3, line = -0.15, cex=0.5)
legend(x = legend_position, legend=c("Male", "Female"),
       col=c("blue", "indianred"), lty = 1, lwd = 3, box.lwd=0)

extra_info = paste("Male: num_points =", length(european_male_column_data),
                   "max =", signif(max(european_male_column_data, na.rm = TRUE), 4),
                   "min =", signif(min(european_male_column_data, na.rm = TRUE), 4), 
                   "median =", signif(median(european_male_column_data, na.rm = TRUE), 4),
                   "mean =", signif(mean(european_male_column_data, na.rm = TRUE), 4),
                   "std =", signif(sd(european_male_column_data, na.rm = TRUE), 4), "\n",
                   "Female: num_points=", length(european_female_column_data),
                   "max =", signif(max(european_female_column_data, na.rm = TRUE), 4),
                   "min =", signif(min(european_female_column_data, na.rm = TRUE), 4), 
                   "median =", signif(median(european_female_column_data, na.rm = TRUE), 4),
                   "mean =", signif(mean(european_female_column_data, na.rm = TRUE), 4),
                   "std =", signif(sd(european_female_column_data, na.rm = TRUE), 4))
title = paste("European density of", column, "indicator")
plot(density(european_male_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust),
     main=title,
     xlim = c(european_density_min_x, european_density_max_x),
     ylim= c(european_density_min_y, european_density_max_y),
     xlab=column, ylab="Density", col="blue", lwd = 1.5, cex.main=0.85, bty="n")
lines(density(european_female_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust),
      col="indianred", lwd = 1.5)
mtext(extra_info, side = 3, line = -0.15, cex=0.5)
legend(x = legend_position, legend=c("Male", "Female"),
       col=c("blue", "indianred"), lty = 1, lwd = 3, box.lwd=0)

extra_info = paste("Male: num_points =", length(central_american_male_column_data),
                   "max =", signif(max(central_american_male_column_data, na.rm = TRUE), 4),
                   "min =", signif(min(central_american_male_column_data, na.rm = TRUE), 4), 
                   "median =", signif(median(central_american_male_column_data, na.rm = TRUE), 4),
                   "mean =", signif(mean(central_american_male_column_data, na.rm = TRUE), 4),
                   "std =", signif(sd(central_american_male_column_data, na.rm = TRUE), 4), "\n",
                   "Female: num_points=", length(central_american_female_column_data),
                   "max =", signif(max(central_american_female_column_data, na.rm = TRUE), 4),
                   "min =", signif(min(central_american_female_column_data, na.rm = TRUE), 4), 
                   "median =", signif(median(central_american_female_column_data, na.rm = TRUE), 4),
                   "mean =", signif(mean(central_american_female_column_data, na.rm = TRUE), 4),
                   "std =", signif(sd(central_american_female_column_data, na.rm = TRUE), 4))
title = paste("Central american density of", column, "indicator")
plot(density(central_american_male_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust),
     main=title,
     xlim = c(central_american_density_min_x, central_american_density_max_x),
     ylim= c(central_american_density_min_y, central_american_density_max_y),
     xlab=column, ylab="Density", col="blue", lwd = 1.5, cex.main=0.85, bty="n")
lines(density(central_american_female_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust),
      col="indianred", lwd = 1.5)
mtext(extra_info, side = 3, line = -0.15, cex=0.5)
legend(x = legend_position, legend=c("Male", "Female"),
       col=c("blue", "indianred"), lty = 1, lwd = 3, box.lwd=0)
dev.off()
}
                                      


# active_days__callandtext__mean
generate_column_dist_plots(column = "active_days__callandtext__mean",
                           density_num_points = 512,
                           south_asian_num_bins = 80,
                           european_num_bins = 80,
                           central_american_num_bins = 80,
                           density_min_x = 3,
                           density_max_x = 7,
                           density_min_y = 0,
                           density_max_y = 2,
                           south_asian_histogram_min_x = 3,
                           south_asian_histogram_max_x = 7,
                           european_histogram_min_x = 3,
                           european_histogram_max_x = 7,
                           central_american_histogram_min_x = 3,
                           central_american_histogram_max_x = 7,
                           south_asian_density_min_x = 3,
                           south_asian_density_max_x = 7,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 2.1,
                           european_density_min_x = 3,
                           european_density_max_x = 7,
                           european_density_min_y = 0,
                           european_density_max_y = 2.5,
                           central_american_density_min_x = 3,
                           central_american_density_max_x = 7,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.9,
                           legend_position = "topleft")




# number_of_contacts__call__mean
generate_column_dist_plots(column = "number_of_contacts__call__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 1000,
                           european_num_bins = 500,
                           central_american_num_bins = 1300,
                           density_min_x = 1,
                           density_max_x = 80,
                           density_min_y = 0,
                           density_max_y = 0.1,
                           south_asian_histogram_min_x = 1,
                           south_asian_histogram_max_x = 80,
                           european_histogram_min_x = 1,
                           european_histogram_max_x = 40,
                           central_american_histogram_min_x = 1,
                           central_american_histogram_max_x = 40,
                           south_asian_density_min_x = 1,
                           south_asian_density_max_x = 80,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.06,
                           european_density_min_x = 1,
                           european_density_max_x = 40,
                           european_density_min_y = 0,
                           european_density_max_y = 0.11,
                           central_american_density_min_x = 1,
                           central_american_density_max_x = 40,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.11,
                           legend_position = "topright")




# number_of_interactions__call__mean
generate_column_dist_plots(column = "number_of_interactions__call__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 800,
                           european_num_bins = 600,
                           central_american_num_bins = 1000,
                           density_min_x = 0,
                           density_max_x = 200,
                           density_min_y = 0,
                           density_max_y = 0.03,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 200,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 100,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 100,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 200, 
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.015,
                           european_density_min_x = 0,
                           european_density_max_x = 100,
                           european_density_min_y = 0,
                           european_density_max_y = 0.03,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 100,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.025,
                           legend_position = "topright")      




# duration_of_calls__call__std__mean
generate_column_dist_plots(column = "duration_of_calls__call__std__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 100,
                           european_num_bins = 500,
                           central_american_num_bins = 300,
                           density_min_x = 0,
                           density_max_x = 1200,
                           density_min_y = 0,
                           density_max_y = 0.008,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 500,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1500,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 300,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 500,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.008,
                           european_density_min_x = 0,
                           european_density_max_x = 1500,
                           european_density_min_y = 0,
                           european_density_max_y = 0.002,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 300,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.01,
                           legend_position = "topright")




# duration_of_calls__call__median__mean
generate_column_dist_plots(column = "duration_of_calls__call__median__mean",
                           density_num_points = 8096,
                           south_asian_num_bins = 400,
                           european_num_bins = 1300,
                           central_american_num_bins = 600,
                           density_min_x = 0,
                           density_max_x = 400,
                           density_min_y = 0,
                           density_max_y = 0.025,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 150,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 500,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 200,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 150,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.03,
                           european_density_min_x = 0,
                           european_density_max_x = 500,
                           european_density_min_y = 0,
                           european_density_max_y = 0.007,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 200,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.022,
                           legend_position = "topright")





# duration_of_calls__call__mean__mean
generate_column_dist_plots(column = "duration_of_calls__call__mean__mean",
                           density_num_points = 512,
                           south_asian_num_bins = 200,
                           european_num_bins = 600,
                           central_american_num_bins = 600,
                           density_min_x = 0,
                           density_max_x = 1000,
                           density_min_y = 0,
                           density_max_y = 0.012,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 400,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1200,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 400,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 400,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.013,
                           european_density_min_x = 0,
                           european_density_max_x = 1200,
                           european_density_min_y = 0,
                           european_density_max_y = 0.003,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 400,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.012,
                           legend_position = "topright")






# percent_nocturnal__call__mean
generate_column_dist_plots(column = "percent_nocturnal__call__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 50,
                           european_num_bins = 50,
                           central_american_num_bins = 50,
                           density_min_x = 0,
                           density_max_x = 1,
                           density_min_y = 0,
                           density_max_y = 4.5,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 1,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 1,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 1,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 4.5,
                           european_density_min_x = 0,
                           european_density_max_x = 1,
                           european_density_min_y = 0,
                           european_density_max_y = 4.5,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 1,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 4,
                           legend_position = "topright")






# percent_initiated_conversation__callandtext__mean
generate_column_dist_plots(column = "percent_initiated_conversation__callandtext__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 50,
                           european_num_bins = 50,
                           central_american_num_bins = 50,
                           density_min_x = 0,
                           density_max_x = 1,
                           density_min_y = 0,
                           density_max_y = 3,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 1,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 1,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 1,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 3,
                           european_density_min_x = 0,
                           european_density_max_x = 1,
                           european_density_min_y = 0,
                           european_density_max_y = 3.5,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 1,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 2.1,
                           legend_position = "topright")





# percent_initiated_interactions__call__mean
generate_column_dist_plots(column = "percent_initiated_interactions__call__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 50,
                           european_num_bins = 50,
                           central_american_num_bins = 50,
                           density_min_x = 0,
                           density_max_x = 1,
                           density_min_y = 0,
                           density_max_y = 3.5,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 1,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 1,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 1,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 2.5,
                           european_density_min_x = 0,
                           european_density_max_x = 1,
                           european_density_min_y = 0,
                           european_density_max_y = 3.5,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 1,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 3,
                           legend_position = "topleft")







# entropy_of_contacts__call__mean
generate_column_dist_plots(column = "entropy_of_contacts__call__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 200,
                           european_num_bins = 100,
                           central_american_num_bins = 100,
                           density_min_x = 0,
                           density_max_x = 4,
                           density_min_y = 0,
                           density_max_y = 0.8,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 4,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 4,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 4,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 4,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.6,
                           european_density_min_x = 0,
                           european_density_max_x = 4,
                           european_density_min_y = 0,
                           european_density_max_y = 0.9,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 4,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.9,
                           legend_position = "topright")





# interactions_per_contact__call__std__mean
generate_column_dist_plots(column = "interactions_per_contact__call__std__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 400,
                           european_num_bins = 600,
                           central_american_num_bins = 800,
                           density_min_x = 0,
                           density_max_x = 12,
                           density_min_y = 0,
                           density_max_y = 0.35,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 12,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 10,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 10,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 12,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.3,
                           european_density_min_x = 0,
                           european_density_max_x = 10,
                           european_density_min_y = 0,
                           european_density_max_y = 0.45,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 10,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.3,
                           legend_position = "topright")





# interactions_per_contact__call__median__mean 
generate_column_dist_plots(column = "interactions_per_contact__call__median__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 2000,
                           european_num_bins = 600,
                           central_american_num_bins = 800,
                           density_min_x = 1,
                           density_max_x = 5,
                           density_min_y = 0,
                           density_max_y = 1.2,
                           south_asian_histogram_min_x = 1,
                           south_asian_histogram_max_x = 5,
                           european_histogram_min_x = 1,
                           european_histogram_max_x = 5,
                           central_american_histogram_min_x = 1,
                           central_american_histogram_max_x = 5,
                           south_asian_density_min_x = 1,
                           south_asian_density_max_x = 5,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 1.2,
                           european_density_min_x = 1,
                           european_density_max_x = 5,
                           european_density_min_y = 0,
                           european_density_max_y = 1,
                           central_american_density_min_x = 1,
                           central_american_density_max_x = 5,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 1.1,
                           legend_position = "topright")





# interactions_per_contact__call__mean__mean 
generate_column_dist_plots(column = "interactions_per_contact__call__mean__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 1000,
                           european_num_bins = 600,
                           central_american_num_bins = 1000,
                           density_min_x = 1,
                           density_max_x = 14,
                           density_min_y = 0,
                           density_max_y = 0.5,
                           south_asian_histogram_min_x = 1,
                           south_asian_histogram_max_x = 8,
                           european_histogram_min_x = 1,
                           european_histogram_max_x = 8,
                           central_american_histogram_min_x = 1,
                           central_american_histogram_max_x = 11,
                           south_asian_density_min_x = 1,
                           south_asian_density_max_x = 8,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.5,
                           european_density_min_x = 1,
                           european_density_max_x = 8,
                           european_density_min_y = 0,
                           european_density_max_y = 0.5,
                           central_american_density_min_x = 1,
                           central_american_density_max_x = 11,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.5,
                           legend_position = "topright")






# interevents_time__call__std__mean
generate_column_dist_plots(column = "interevents_time__call__std__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 100,
                           european_num_bins = 300,
                           central_american_num_bins = 200,
                           density_min_x = 0,
                           density_max_x = 100000,
                           density_min_y = 0,
                           density_max_y = 0.00004,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 80000,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 80000,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 80000,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 80000,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.00004,
                           european_density_min_x = 0,
                           european_density_max_x = 80000,
                           european_density_min_y = 0,
                           european_density_max_y = 0.00003,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 80000,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.000035,
                           legend_position = "topright")





# interevents_time__call__median__mean
generate_column_dist_plots(column = "interevents_time__call__median__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 500,
                           european_num_bins = 300,
                           central_american_num_bins = 400,
                           density_min_x = 0,
                           density_max_x = 60000,
                           density_min_y = 0,
                           density_max_y = 0.00016,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 30000,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 60000,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 40000,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 30000,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.00017,
                           european_density_min_x = 0,
                           european_density_max_x = 60000,
                           european_density_min_y = 0,
                           european_density_max_y = 0.00009,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 40000,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.00011,
                           legend_position = "topright")





# interevents_time__call__mean__mean
generate_column_dist_plots(column = "interevents_time__call__mean__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 200,
                           european_num_bins = 200,
                           central_american_num_bins = 200,
                           density_min_x = 0,
                           density_max_x = 100000,
                           density_min_y = 0,
                           density_max_y = 0.000055,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 80000,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 100000,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 80000,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 80000,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.00006,
                           european_density_min_x = 0,
                           european_density_max_x = 100000,
                           european_density_min_y = 0,
                           european_density_max_y = 0.000035,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 80000,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.00004,
                           legend_position = "topright")





# number_of_places__mean
generate_column_dist_plots(column = "number_of_places__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 300,
                           european_num_bins = 200,
                           central_american_num_bins = 200,
                           density_min_x = 0,
                           density_max_x = 50,
                           density_min_y = 0,
                           density_max_y = 0.14,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 30,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 60,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 25,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 30,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.17,
                           european_density_min_x = 0,
                           european_density_max_x = 60,
                           european_density_min_y = 0,
                           european_density_max_y = 0.05,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 25,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.15,
                           legend_position = "topright")





# entropy_places__mean
generate_column_dist_plots(column = "entropy_places__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 100,
                           european_num_bins = 100,
                           central_american_num_bins = 100,
                           density_min_x = 0,
                           density_max_x = 4,
                           density_min_y = 0,
                           density_max_y = 1,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 4,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 4,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 4,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 4,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.9,
                           european_density_min_x = 0,
                           european_density_max_x = 4,
                           european_density_min_y = 0,
                           european_density_max_y = 0.7,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 4,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 1,
                           legend_position = "topright")





# percent_at_home__mean
generate_column_dist_plots(column = "percent_at_home__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 50,
                           european_num_bins = 50,
                           central_american_num_bins = 50,
                           density_min_x = 0,
                           density_max_x = 1,
                           density_min_y = 0,
                           density_max_y = 2.5,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 1,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 1,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 1,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 2,
                           european_density_min_x = 0,
                           european_density_max_x = 1,
                           european_density_min_y = 0,
                           european_density_max_y = 3,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 1,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 2.5,
                           legend_position = "topright")





# radius_of_gyration__mean
generate_column_dist_plots(column = "radius_of_gyration__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 300,
                           european_num_bins = 400,
                           central_american_num_bins = 800,
                           density_min_x = 0,
                           density_max_x = 100,
                           density_min_y = 0,
                           density_max_y = 0.12,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 30,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 120,
                           central_american_histogram_min_x = 1,
                           central_american_histogram_max_x = 100,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 30,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.18,
                           european_density_min_x = 0,
                           european_density_max_x = 120,
                           european_density_min_y = 0,
                           european_density_max_y = 0.035,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 100,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.08,
                           legend_position = "topright")





# number_of_contacts__text__mean
generate_column_dist_plots(column = "number_of_contacts__text__mean",
                           density_num_points = 4096,
                           south_asian_num_bins = 1500,
                           european_num_bins = 700,
                           central_american_num_bins = 6000,
                           density_min_x = 1,
                           density_max_x = 30,
                           density_min_y = 0,
                           density_max_y = 0.35,
                           south_asian_histogram_min_x = 1,
                           south_asian_histogram_max_x = 5,
                           european_histogram_min_x = 1,
                           european_histogram_max_x = 30,
                           central_american_histogram_min_x = 1,
                           central_american_histogram_max_x = 10,
                           south_asian_density_min_x = 1,
                           south_asian_density_max_x = 5,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 1.2,
                           european_density_min_x = 1,
                           european_density_max_x = 30,
                           european_density_min_y = 0,
                           european_density_max_y = 0.1,
                           central_american_density_min_x = 1,
                           central_american_density_max_x = 10,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.32,
                           legend_position = "topright")





# number_of_interactions__text__mean
generate_column_dist_plots(column = "number_of_interactions__text__mean",
                           density_num_points = 8096,
                           south_asian_num_bins = 4500,
                           european_num_bins = 2000,
                           central_american_num_bins = 2000,
                           density_min_x = 1,
                           density_max_x = 150,
                           density_min_y = 0,
                           density_max_y = 0.10,
                           south_asian_histogram_min_x = 1,
                           south_asian_histogram_max_x = 20,
                           european_histogram_min_x = 1,
                           european_histogram_max_x = 200,
                           central_american_histogram_min_x = 1,
                           central_american_histogram_max_x = 50,
                           south_asian_density_min_x = 1,
                           south_asian_density_max_x = 20,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.25,
                           european_density_min_x = 1,
                           european_density_max_x = 200,
                           european_density_min_y = 0,
                           european_density_max_y = 0.016,
                           central_american_density_min_x = 1,
                           central_american_density_max_x = 50,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.07,
                           legend_position = "topright")





# percent_nocturnal__text__mean
generate_column_dist_plots(column = "percent_nocturnal__text__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 50,
                           european_num_bins = 50,
                           central_american_num_bins = 50,
                           density_min_x = 0,
                           density_max_x = 1,
                           density_min_y = 0,
                           density_max_y = 4.5,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 1,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 1,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 1,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 2.5,
                           european_density_min_x = 0,
                           european_density_max_x = 1,
                           european_density_min_y = 0,
                           european_density_max_y = 5,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 1,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 3,
                           legend_position = "topright")





# response_rate_text__callandtext__mean
generate_column_dist_plots(column = "response_rate_text__callandtext__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 100,
                           european_num_bins = 100,
                           central_american_num_bins = 100,
                           density_min_x = 0,
                           density_max_x = 1,
                           density_min_y = 0,
                           density_max_y = 5,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 1,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 1,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 1,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 8,
                           european_density_min_x = 0,
                           european_density_max_x = 1,
                           european_density_min_y = 0,
                           european_density_max_y = 3,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 1,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 8,
                           legend_position = "topright")





# entropy_of_contacts__text__mean
generate_column_dist_plots(column = "entropy_of_contacts__text__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 100,
                           european_num_bins = 100,
                           central_american_num_bins = 200,
                           density_min_x = 0,
                           density_max_x = 4,
                           density_min_y = 0,
                           density_max_y = 2.3,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 4,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 4,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 4,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 4,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 2.3,
                           european_density_min_x = 0,
                           european_density_max_x = 4,
                           european_density_min_y = 0,
                           european_density_max_y = 1,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 4,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 1,
                           legend_position = "topright")





# interactions_per_contact__text__std__mean
generate_column_dist_plots(column = "interactions_per_contact__text__std__mean",
                           density_num_points = 4096, 
                           south_asian_num_bins = 4500,
                           european_num_bins = 2000,
                           central_american_num_bins = 1200,
                           density_min_x = 0,
                           density_max_x = 30,
                           density_min_y = 0,
                           density_max_y = 0.6,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 6,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 30,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 10,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 6,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 1.8,
                           european_density_min_x = 0,
                           european_density_max_x = 30,
                           european_density_min_y = 0,
                           european_density_max_y = 0.15,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 10,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.32,
                           legend_position = "topright")





# interactions_per_contact__text__median__mean
generate_column_dist_plots(column = "interactions_per_contact__text__median__mean",
                           density_num_points = 4096, 
                           south_asian_num_bins = 2500,
                           european_num_bins = 700,
                           central_american_num_bins = 800,
                           density_min_x = 1,
                           density_max_x = 10,
                           density_min_y = 0,
                           density_max_y = 0.8,
                           south_asian_histogram_min_x = 1,
                           south_asian_histogram_max_x = 6,
                           european_histogram_min_x = 1,
                           european_histogram_max_x = 10,
                           central_american_histogram_min_x = 1,
                           central_american_histogram_max_x = 10,
                           south_asian_density_min_x = 1,
                           south_asian_density_max_x = 8,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.8,
                           european_density_min_x = 1,
                           european_density_max_x = 10,
                           european_density_min_y = 0,
                           european_density_max_y = 0.5,
                           central_american_density_min_x = 1,
                           central_american_density_max_x = 10,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.4,
                           legend_position = "topright")





# interactions_per_contact__text__mean__mean
generate_column_dist_plots(column = "interactions_per_contact__text__mean__mean",
                           density_num_points = 4096, 
                           south_asian_num_bins = 2500,
                           european_num_bins = 2500,
                           central_american_num_bins = 1000,
                           density_min_x = 1,
                           density_max_x = 15,
                           density_min_y = 0,
                           density_max_y = 0.6,
                           south_asian_histogram_min_x = 1,
                           south_asian_histogram_max_x = 8,
                           european_histogram_min_x = 1,
                           european_histogram_max_x = 15,
                           central_american_histogram_min_x = 1,
                           central_american_histogram_max_x = 14,
                           south_asian_density_min_x = 1,
                           south_asian_density_max_x = 8,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.6,
                           european_density_min_x = 1,
                           european_density_max_x = 15,
                           european_density_min_y = 0,
                           european_density_max_y = 0.2,
                           central_american_density_min_x = 1,
                           central_american_density_max_x = 15,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.25,
                           legend_position = "topright")





# interevents_time__text__std__mean 
generate_column_dist_plots(column = "interevents_time__text__std__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 300,
                           european_num_bins = 100,
                           central_american_num_bins = 300,
                           density_min_x = 0,
                           density_max_x = 80000,
                           density_min_y = 0,
                           density_max_y = 0.000055,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 80000,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 80000,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 80000,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 80000,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.00005,
                           european_density_min_x = 0,
                           european_density_max_x = 80000,
                           european_density_min_y = 0,
                           european_density_max_y = 0.00004,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 80000,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.00003,
                           legend_position = "topright")





# interevents_time__text__median__mean
generate_column_dist_plots(column = "interevents_time__text__median__mean",
                           density_num_points = 2048,
                           south_asian_num_bins = 450,
                           european_num_bins = 800,
                           central_american_num_bins = 400,
                           density_min_x = 0,
                           density_max_x = 80000,
                           density_min_y = 0,
                           density_max_y = 0.0002,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 100000,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 10000,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 80000,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 100000,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.00004,
                           european_density_min_x = 0,
                           european_density_max_x = 10000,
                           european_density_min_y = 0,
                           european_density_max_y = 0.001,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 80000,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.00005,
                           legend_position = "topright")





# interevents_time__text__mean__mean
generate_column_dist_plots(column = "interevents_time__text__mean__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 450,
                           european_num_bins = 200,
                           central_american_num_bins = 300,
                           density_min_x = 0,
                           density_max_x = 100000,
                           density_min_y = 0,
                           density_max_y = 0.00006,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 60000,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 60000,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 110000,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 60000,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.000025,
                           european_density_min_x = 0,
                           european_density_max_x = 60000,
                           european_density_min_y = 0,
                           european_density_max_y = 0.00007,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 110000,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.000022,
                           legend_position = "topright")





# response_delay_text__callandtext__median__mean
generate_column_dist_plots(column = "response_delay_text__callandtext__median__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 100,
                           european_num_bins = 150,
                           central_american_num_bins = 130,
                           density_min_x = 0,
                           density_max_x = 1500,
                           density_min_y = 0,
                           density_max_y = 0.004,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 2000,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1000,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 1500,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 2000,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.0015,
                           european_density_min_x = 0,
                           european_density_max_x = 1000,
                           european_density_min_y = 0,
                           european_density_max_y = 0.004,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 1500,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.0025,
                           legend_position = "topright")





# response_delay_text__callandtext__mean__mean 
generate_column_dist_plots(column = "response_delay_text__callandtext__mean__mean",
                           density_num_points = 1024,
                           south_asian_num_bins = 100,
                           european_num_bins = 150,
                           central_american_num_bins = 100,
                           density_min_x = 0,
                           density_max_x = 2000,
                           density_min_y = 0,
                           density_max_y = 0.0022,
                           south_asian_histogram_min_x = 0,
                           south_asian_histogram_max_x = 2500,
                           european_histogram_min_x = 0,
                           european_histogram_max_x = 1500,
                           central_american_histogram_min_x = 0,
                           central_american_histogram_max_x = 2000,
                           south_asian_density_min_x = 0,
                           south_asian_density_max_x = 2500,
                           south_asian_density_min_y = 0,
                           south_asian_density_max_y = 0.0013,
                           european_density_min_x = 0,
                           european_density_max_x = 1500,
                           european_density_min_y = 0,
                           european_density_max_y = 0.0025,
                           central_american_density_min_x = 0,
                           central_american_density_max_x = 2000,
                           central_american_density_min_y = 0,
                           central_american_density_max_y = 0.0015,
                           legend_position = "topright")
