# add alpha channel to colors
svm_l_color_code = "#E5DD3B"
svm_l_alpha = 0.5
svm_k_color_code = "#B01E1B"
svm_k_alpha = 1
rf_color_code = "#32567F"
rf_alpha = 0.5
logistic_color_code = "#59FFB1"
logistic_alpha = 0.5
knn_color_code = "#C07737"
knn_alpha = 0.5
svm_l_color = col2rgb(svm_l_color_code)/255
svm_l_color = rgb(red = svm_l_color[1], green = svm_l_color[2], blue = svm_l_color[3], alpha = svm_l_alpha)
svm_k_color = col2rgb(svm_k_color_code)/255
svm_k_color = rgb(red = svm_k_color[1], green = svm_k_color[2], blue = svm_k_color[3], alpha = svm_k_alpha)
rf_color = col2rgb(rf_color_code)/255
rf_color = rgb(red = rf_color[1], green = rf_color[2], blue = rf_color[3], alpha = rf_alpha)
logistic_color = col2rgb(logistic_color_code)/255
logistic_color = rgb(red = logistic_color[1], green = logistic_color[2], blue = logistic_color[3], alpha = logistic_alpha)
knn_color = col2rgb(knn_color_code)/255
knn_color = rgb(red = knn_color[1], green = knn_color[2], blue = knn_color[3], alpha = knn_alpha)

generate_subset_accuracy_plots = function(filename,
                                          data,
                                          svm_l,
                                          svm_k,
                                          rf,
                                          logistic,
                                          knn,
                                          plot_margins,
                                          title,
                                          xlab,
                                          ylab,
                                          x_min,
                                          x_max,
                                          y_min,
                                          y_max,
                                          plot_points,
                                          pch,
                                          lwd,
                                          loess.span,
                                          xticks,
                                          xlabels,
                                          x.padj,
                                          xlab.margin,
                                          yticks,
                                          ylabels,
                                          y.padj,
                                          ylab.margin,
                                          axis.lwd,
                                          draw_legend,
                                          legend_position,
                                          font.main,
                                          cex.pt,
                                          cex.axis,
                                          cex.main,
                                          cex.lab,
                                          cex.legend) {
  pdf(filename, width=20, height=20, pointsize = 25)
  par(mar = plot_margins)

  legends = character(0)
  legend_colors = character(0)

  #percentage_predicted = data$Percentage.predicted[data$Percentage.predicted > threshold]
  #svm_l_data = data$SVM.L[data$Percentage.predicted > threshold]
  #svm_k_data = data$SVM.K[data$Percentage.predicted > threshold]
  #rf_data = data$RF[data$Percentage.predicted > threshold]
  #logistic_data = data$LOGISTIC[data$Percentage.predicted > threshold]
  #knn_data = data$KNN[data$Percentage.predicted > threshold]
  percentage_predicted = data$Percentage.predicted
  svm_l_data = data$SVM.L
  svm_k_data = data$SVM.K
  rf_data = data$RF
  logistic_data = data$LOGISTIC
  knn_data = data$KNN

  plot(1,
       type="n",
       main = "",
       xlab="",
       ylab="",
       xlim=c(x_min, x_max),
       ylim=c(y_min, y_max),
       xaxt = "n",
       yaxt = "n",
       xaxs = "i",
       yaxs = "i",
       font.main = font.main)

  
  ordered = order(percentage_predicted)
  if (svm_l) {
    loess_fit = loess(svm_l_data ~ percentage_predicted, span = loess.span)
    lines(percentage_predicted[ordered], loess_fit$fitted[ordered], col=svm_l_color, lwd=lwd)
    if (plot_points) {
      points(percentage_predicted, svm_l_data,
             cex = cex.pt,
             pch = pch,  
             col = svm_l_color)
    }
    legends = append(legends, c("SVM-Linear"))
    legend_colors = append(legend_colors, c(svm_l_color))
  }
  
  if (svm_k) {
    loess_fit = loess(svm_k_data ~ percentage_predicted, span = loess.span)
    lines(percentage_predicted[ordered], loess_fit$fitted[ordered], col=svm_k_color, lwd=lwd)
    if (plot_points) {
      points(percentage_predicted, svm_k_data,
             cex = cex.pt,
             pch = pch,  
             col = svm_k_color)
    }
    legends = append(legends, c("SVM-RBF"))
    legend_colors = append(legend_colors, c(svm_k_color))
  }
  
  if (rf) {
    loess_fit = loess(rf_data ~ percentage_predicted, span = loess.span)
    lines(percentage_predicted[ordered], loess_fit$fitted[ordered], col=rf_color, lwd=lwd)
    if (plot_points) {
      points(percentage_predicted, rf_data,
             cex = cex.pt,
             pch = pch,  
             col = rf_color)
    }
    legends = append(legends, c("RF"))
    legend_colors = append(legend_colors, c(rf_color))
  }
  
  if (logistic) {
    loess_fit = loess(logistic_data ~ percentage_predicted, span = loess.span)
    lines(percentage_predicted[ordered], loess_fit$fitted[ordered], col=logistic_color, lwd=lwd)
    if (plot_points) {
      points(percentage_predicted, logistic_data,
             cex = cex.pt,
             pch = pch,  
             col = logistic_color)
    }
    legends = append(legends, c("Logistic"))
    legend_colors = append(legend_colors, c(logistic_color))
  }

  if (knn) {
    loess_fit = loess(knn_data ~ percentage_predicted, span = loess.span)
    lines(percentage_predicted[ordered], loess_fit$fitted[ordered], col=knn_color, lwd=lwd)
    if (plot_points) {
      points(percentage_predicted, kn_data,
             cex = cex.pt,
             pch = pch,  
             col = knn_color)
    }
    legends = append(legends, c("KNN"))
    legend_colors = append(legend_colors, c(knn_color))
  }
  
  axis(side = 1, lwd = axis.lwd,
       at = xticks, labels = xlabels,
       cex.axis = cex.axis, cex.lab = cex.lab, padj=x.padj)
  axis(side = 2, lwd = axis.lwd,
       at = yticks, labels = ylabels,
       cex.axis = cex.axis, cex.lab = cex.lab, padj=y.padj)

  title(main = title, cex.main = cex.main, font = font.main)
  title(xlab = xlab, cex.lab = cex.lab, mgp = c(xlab.margin,1,0), font = font.main)
  title(ylab = ylab, cex.lab = cex.lab, mgp = c(ylab.margin,1,0), font = font.main)

  box(which="plot", bty="o", lwd = axis.lwd)

  if (draw_legend) {
    points = c()
    if (plot_points) {
      points = c(pch, pch, pch, pch, pch)
    }
    legend(x = legend_position,
           legend=legends,
           col=legend_colors,
           pch=points,
           lty = 1,
           lwd = lwd,
           cex=cex.legend,
           pt.cex=cex.pt,
           bty="n")
  }
  dev.off()
}  

#################################

eu_file_suffixes = c("trs_5000.tes_15000",
                     "trs_10000.tes_15000",
                     "trs_20000.tes_10000",
                     "trs_-1.tes_10000")

for (file_suffix in eu_file_suffixes) {
  setwd(dir = "~/research/gender_prediction/results/extended_indicators/subset_stats/full-accuracy/")
  eu_full_accuracy_file = paste("eu-full-accuracy.", file_suffix, ".csv", sep="")
  eu_full_accuracy_data = read.csv(file = eu_full_accuracy_file, header = TRUE, sep = ",")
  
  # set the dir for output plots
  setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators/subset-accuracy/")
  output_filename = paste("eu-full-accuracy.", file_suffix, ".pdf", sep="")
  generate_subset_accuracy_plots(filename = output_filename,
                                 eu_full_accuracy_data,
                                 svm_l = TRUE,
                                 svm_k = TRUE,
                                 rf = TRUE,
                                 logistic = TRUE,
                                 knn = TRUE,
                                 plot_margins = c(7,6.2,2.5,2.5),
                                 title = "",
                                 xlab = "Most Confident",
                                 ylab = "Accuracy (%)",
                                 x_min = 25,
                                 x_max = 100,
                                 y_min = 65,
                                 y_max = 95,
                                 plot_points = FALSE,
                                 pch = 19,
                                 lwd = 18,
                                 loess.span = 0.1,
                                 xticks = c(25,50,75,100),
                                 xlabels = c("25", "50", "75", "100"),
                                 x.padj = 0.6,
                                 xlab.margin = 4.5,
                                 yticks = c(65,75,85,95),
                                 ylabels = c("65","75","85","95"),
                                 y.padj = -0.2,
                                 ylab.margin = 4,
                                 axis.lwd = 11,
                                 draw_legend = TRUE,
                                 legend_position = "topright",
                                 font.main = 1,
                                 cex.pt = 1,
                                 cex.axis = 3.3,
                                 cex.main = 2.5,
                                 cex.lab = 3,
                                 cex.legend = 2.5)
}

sa_file_suffixes = c("trs_5000.tes_15000",
                     "trs_5000.tes_15000.scikitbalancing",
                     "trs_10000.tes_15000",
                     "trs_10000.tes_15000.scikitbalancing",
                     "trs_20000.tes_5000",
                     "trs_-1.tes_5000")
for (file_suffix in sa_file_suffixes) {
  setwd(dir = "~/research/gender_prediction/results/extended_indicators/subset_stats/full-accuracy/")
  sa_full_accuracy_file = paste("sa-full-accuracy.", file_suffix, ".csv", sep="")
  sa_full_accuracy_data = read.csv(file = sa_full_accuracy_file, header = TRUE, sep = ",")
  
  # set the dir for output plots
  setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators/subset-accuracy/")
  output_filename = paste("sa-full-accuracy_", file_suffix, ".pdf", sep="")
  generate_subset_accuracy_plots(filename = output_filename,
                                 sa_full_accuracy_data,
                                 svm_l = TRUE,
                                 svm_k = TRUE,
                                 rf = TRUE,
                                 logistic = TRUE,
                                 knn = TRUE,
                                 plot_margins = c(7,6.2,2.5,2.5),
                                 title = "",
                                 xlab = "Most Confident",
                                 ylab = "Accuracy (%)",
                                 x_min = 25,
                                 x_max = 100,
                                 y_min = 65,
                                 y_max = 95,
                                 plot_points = FALSE,
                                 pch = 19,
                                 lwd = 18,
                                 loess.span = 0.1,
                                 xticks = c(25,50,75,100),
                                 xlabels = c("25", "50", "75", "100"),
                                 x.padj = 0.6,
                                 xlab.margin = 4.5,
                                 yticks = c(65,75,85,95),
                                 ylabels = c("65","75","85","95"),
                                 y.padj = -0.2,
                                 ylab.margin = 4,
                                 axis.lwd = 11,
                                 draw_legend = TRUE,
                                 legend_position = "topright",
                                 font.main = 1,
                                 cex.pt = 1,
                                 cex.axis = 3.3,
                                 cex.main = 2.5,
                                 cex.lab = 3,
                                 cex.legend = 2.5)
  
}