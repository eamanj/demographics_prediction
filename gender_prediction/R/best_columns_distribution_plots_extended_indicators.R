
generate_column_dist_plots = function(country,
                                      column,
                                      title,
                                      xlab,
                                      xticks,
                                      xlabels,
                                      ylab,
                                      yticks,
                                      ylabels,
                                      density_num_points,
                                      density_adjust,
                                      density_min_x,
                                      density_max_x,
                                      density_min_y,
                                      density_max_y,
                                      legend_position,
                                      male_color,
                                      female_color,
                                      lwd,
                                      axis.lwd,
                                      plot_margins,
                                      x.padj,
                                      y.padj,
                                      cex.axis,
                                      cex.lab,
                                      cex.main,
                                      font.main,
                                      cex.legend) {
    filename=paste0(country, "_", column, ".pdf")
    pdf(filename, width=20, height=20, pointsize = 25)
    par(mar = plot_margins)
    #options(scipen=1) 
    
    column_data = data[[column]]
    male_column_data = male_data[[column]]
    female_column_data = female_data[[column]]
    
    male_density = density(male_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust)
    male_density_max_x = male_density$x[which.max(male_density$y)]
    male_density_max_y = male_density$y[which.max(male_density$y)]
    female_density = density(female_column_data, na.rm=TRUE, n=density_num_points, adjust=density_adjust)
    female_density_max_x = female_density$x[which.max(female_density$y)]
    female_density_max_y = female_density$y[which.max(female_density$y)]
    print(paste('Male density max x', male_density_max_x))
    print(paste('Female density max x', female_density_max_x))
    plot(male_density,
         main = "",
         xlab = "",
         ylab = "",
         xlim = c(density_min_x, density_max_x),
         ylim= c(density_min_y, density_max_y),
         xaxt = "n",
         yaxt = "n",
         col=male_color,
         cex.main=cex.main,
         font.main=font.main,
         lwd = lwd,
         xaxs = "i",
         yaxs = "i")
    lines(female_density, col=female_color, lwd = lwd)
    segments(x0 = male_density_max_x, y0 = 0, x1 = male_density_max_x, y1 = male_density_max_y,
             col = male_color, lty = 2, lwd = axis.lwd)
    segments(x0 = female_density_max_x, y0 = 0, x1 = female_density_max_x, y1 = female_density_max_y,
             col = female_color, lty = 2, lwd = axis.lwd)
    
    axis(side = 1, lwd = axis.lwd,
         at = xticks, labels = xlabels,
         cex.axis = cex.axis, cex.lab = cex.lab, padj=x.padj)
    #axis(side = 1, lwd = axis.lwd,
    #     at = female_density_max_x, labels = as.character(round(female_density_max_x, 2)),
    #     col.axis = female_color, col.ticks = female_color,
    #     cex.axis = cex.axis, padj=x.padj)
    #axis(side = 1, lwd = axis.lwd,
    #     at = male_density_max_x, labels = as.character(round(male_density_max_x, 2)),
    #     col.axis = male_color, col.ticks = male_color,
    #     cex.axis = cex.axis, padj=x.padj)
    axis(side = 2, lwd = axis.lwd,
         at = yticks, labels = ylabels,
         cex.axis = cex.axis, cex.lab = cex.lab, padj=y.padj)

    title(main = title, xlab = xlab, ylab = ylab, cex.main = cex.main, cex.lab = cex.lab,
          mgp = c(4,1,0))
    box(which="plot", bty="o", lwd = axis.lwd)
    legend(x = legend_position, legend=c("Male", "Female"),
           col=c(male_color, female_color), lty = 1, lwd = lwd, bty="n", cex=cex.legend)
    
    dev.off()
}


###########################################

# South Asian data
setwd(dir = "~/research/gender_prediction/data/extended_indicators/south_asian/")
data = read.csv(file = "cleaned_south_asian_data_a2_c2_without_median_skewness_kurtosis.csv", header = TRUE, sep = ",")
male_data = data[data$attributes__surv_gender == 1,]
female_data = data[data$attributes__surv_gender == 0,]
male_color = "#FF4DFF"
female_color = "#95FF5C"

setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators")
generate_column_dist_plots(country = "south_asian",
                           column = "percent_initiated_interactions__weekend__night__call__mean",
                           title = "",
                           xlab = "[%]",
                           xticks = c(0,0.2,0.4,0.6,0.8,1),
                           xlabels = c("0", "20", "40", "60", "80", "100"),
                           ylab = "",
                           yticks = c(0,0.5,1,1.5,2,2.5),
                           ylabels = c("0","0.5","1","1.5","2","2.5"),
                           density_num_points = 512,
                           density_adjust = 1.2,
                           density_min_x = 0,
                           density_max_x = 1,
                           density_min_y = 0,
                           density_max_y = 2.5,
                           legend_position = "topright",
                           male_color,
                           female_color,
                           lwd = 12,
                           axis.lwd = 11,
                           plot_margins = c(6,4,4,2.2),
                           x.padj = 0.5,
                           y.padj = 0,
                           cex.axis = 3.3,
                           cex.lab = 3.3,
                           cex.main = 2.5,
                           font.main = 1,
                           cex.legend = 3)

generate_column_dist_plots(country = "south_asian",
                           column = "percent_at_home__weekend__day__mean",
                           title = "% at Home Weekend Day",
                           xlab = "",
                           xticks = c(0,0.2,0.4,0.6,0.8,1),
                           xlabels = c("0", "0.2", "0.4", "", "0.8", "1"),
                           ylab = "",
                           yticks = c(0,0.5,1,1.5),
                           ylabels = c("0","0.5","1","1.5"),
                           density_num_points = 512,
                           density_adjust = 1.2,
                           density_min_x = 0,
                           density_max_x = 1,
                           density_min_y = 0,
                           density_max_y = 1.75,
                           legend_position = "topright",
                           male_color,
                           female_color,
                           lwd = 12,
                           axis.lwd = 11,
                           plot_margins = c(6,4,4,2),
                           x.padj = 0.5,
                           y.padj = 0,
                           cex.axis = 3.3,
                           cex.lab = 3.5,
                           cex.main = 2.5,
                           font.main = 1,
                           cex.legend = 3)

# European data
setwd(dir = "~/research/gender_prediction/data/extended_indicators/european/")
data = read.csv(file = "cleaned_european_data_a2_c2_top200.csv.20%sample", header = TRUE, sep = ",")
male_data = data[data$attributes__survey_gender == 1,]
female_data = data[data$attributes__survey_gender== 0,]
male_color = "#B217B2"
female_color = "#50B21A"

setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators")
generate_column_dist_plots(country = "european",
                           column = "response_delay_text__weekday__allday__callandtext__max__mean",
                           title = "",
                           xlab = "[min]",
                           xticks = c(0,1200,2400,3600),
                           xlabels = c("0","20","40","60"),
                           ylab = "",
                           yticks = c(0,0.0002,0.0004,0.0006),
                           ylabels = c("0","2e-3","4e-3","6e-3"),
                           density_num_points = 512,
                           density_adjust = 1.2,
                           density_min_x = 0,
                           density_max_x = 3600,
                           density_min_y = 0,
                           density_max_y = 0.0006,
                           legend_position = "topright",
                           male_color,
                           female_color,
                           lwd = 12,
                           axis.lwd = 11,
                           plot_margins = c(6,4,4,2),
                           x.padj = 0.5,
                           y.padj = 0,
                           cex.axis = 3.3,
                           cex.lab = 3.5,
                           cex.main = 2.5,
                           font.main = 1,
                           cex.legend = 3)


generate_column_dist_plots(country = "european",
                           column = "percent_initiated_conversations__weekend__day__callandtext__mean",
                           title = "% Initiated Conversations Weekend Day Call and Text",
                           xlab = "",
                           xticks = c(0,0.2,0.4,0.6,0.8,1),
                           xlabels = c("0", "0.2", "0.4", "0.6", "0.8", "1"),
                           ylab = "",
                           yticks = c(0,0.5,1,1.5,2,2.5),
                           ylabels = c("0","0.5","1","1.5","2","2.5"),
                           density_num_points = 512,
                           density_adjust = 1.5,
                           density_min_x = 0,
                           density_max_x = 1,
                           density_min_y = 0,
                           density_max_y = 2.8,
                           legend_position = "topright",
                           male_color,
                           female_color,
                           lwd = 12,
                           axis.lwd = 11,
                           plot_margins = c(6,4,4,2),
                           x.padj = 0.5,
                           y.padj = 0,
                           cex.axis = 3.3,
                           cex.lab = 3.5,
                           cex.main = 2.5,
                           font.main = 1,
                           cex.legend = 3)
