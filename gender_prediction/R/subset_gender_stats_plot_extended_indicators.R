male_color = "#B217B2"
female_color = "#50B21A"
  
generate_subset_precision_plots = function(filename,
                                           data,
                                           method,
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

  if (method == "svm_l") {
    male_percent_predicted = data$Male.SVM.L.percent.predicted
    male_precision = data$Male.SVM.L.Precision
    
    female_percent_predicted = data$Female.SVM.L.percent.predicted
    female_precision = data$Female.SVM.L.Precision
    legends = c("Male-SVM-Linear", "Female-SVM-Linear")
  } else if (method == "svm_k") {
    male_percent_predicted = data$Male.SVM.K.percent.predicted
    male_precision = data$Male.SVM.K.Precision
    
    female_percent_predicted = data$Female.SVM.K.percent.predicted
    female_precision = data$Female.SVM.K.Precision
    legends = c("Male-SVM-RBF", "Female-SVM-RBF")
  } else if (method == "rf") {
    male_percent_predicted = data$Male.RF.percent.predicted
    male_precision = data$Male.RF.Precision
    
    female_percent_predicted = data$Female.RF.percent.predicted
    female_precision = data$Female.RF.Precision
    legends = c("Male-RF", "Female-RF")
  } else if (method == "logistic") {
    male_percent_predicted = data$Male.LOGISTIC.percent.predicted
    male_precision = data$Male.LOGISTIC.Precision
    
    female_percent_predicted = data$Female.LOGISTIC.percent.predicted
    female_precision = data$Female.LOGISTIC.Precision
    legends = c("Male-Logistic", "Female-Logistic")
  } else if (method == "knn") {
    male_percent_predicted = data$Male.KNN.percent.predicted
    male_precision = data$Male.KNN.Precision
    
    female_percent_predicted = data$Female.KNN.percent.predicted
    female_precision = data$Female.KNN.Precision
    legends = c("Male-KNN", "Female-KNN")
  } else {
    print('Bad Method')
    return
  }

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
  
  male_ordered = order(male_percent_predicted)
  female_ordered = order(female_percent_predicted)
  male_loess_fit = loess(male_precision ~ male_percent_predicted, span = loess.span)
  female_loess_fit = loess(female_precision ~ female_percent_predicted, span = loess.span)
  lines(male_percent_predicted[male_ordered], male_loess_fit$fitted[male_ordered], col=male_color, lwd=lwd)
  lines(female_percent_predicted[female_ordered], female_loess_fit$fitted[female_ordered], col=female_color, lwd=lwd)
  if (plot_points) {
    points(male_percent_predicted, male_precision,
           cex = cex.pt,
           pch = pch,  
           col = male_color)
    points(female_percent_predicted, female_precision,
           cex = cex.pt,
           pch = pch,  
           col = female_color)
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

  if (draw_legend) {
    points = c()
    if (plot_points) {
      points = c(pch, pch)
    }
    legend(x = legend_position,
           legend=legends,
           col=c(male_color, female_color),
           pch=points,
           lty = 1,
           lwd = lwd,
           cex=cex.legend,
           pt.cex=cex.pt,
           bty="n")
  }

  box(which="plot", bty="o", lwd = axis.lwd)
  dev.off()
}  

generate_subset_recall_plots = function(filename,
                                        data,
                                        method,
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

  if (method == "svm_l") {
    male_percent_predicted = data$Male.SVM.L.percent.predicted
    male_recall = data$Male.SVM.L.Recall
    
    female_percent_predicted = data$Female.SVM.L.percent.predicted
    female_recall = data$Female.SVM.L.Recall
    legends = c("Male-SVM-Linear", "Female-SVM-Linear")
  } else if (method == "svm_k") {
    male_percent_predicted = data$Male.SVM.K.percent.predicted
    male_recall = data$Male.SVM.K.Recall
    
    female_percent_predicted = data$Female.SVM.K.percent.predicted
    female_recall = data$Female.SVM.K.Recall
    legends = c("Male-SVM-RBF", "Female-SVM-RBF")
  } else if (method == "rf") {
    male_percent_predicted = data$Male.RF.percent.predicted
    male_recall = data$Male.RF.Recall
    
    female_percent_predicted = data$Female.RF.percent.predicted
    female_recall = data$Female.RF.Recall
    legends = c("Male-RF", "Female-RF")
  } else if (method == "logistic") {
    male_percent_predicted = data$Male.LOGISTIC.percent.predicted
    male_recall = data$Male.LOGISTIC.Recall
    
    female_percent_predicted = data$Female.LOGISTIC.percent.predicted
    female_recall = data$Female.LOGISTIC.Recall
    legends = c("Male-Logistic", "Female-Logistic")
  } else if (method == "knn") {
    male_percent_predicted = data$Male.KNN.percent.predicted
    male_recall = data$Male.KNN.Recall
    
    female_percent_predicted = data$Female.KNN.percent.predicted
    female_recall = data$Female.KNN.Recall
    legends = c("Male-KNN", "Female-KNN")
  } else {
    print('Bad Method')
    return
  }

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
  
  male_ordered = order(male_percent_predicted)
  female_ordered = order(female_percent_predicted)
  male_loess_fit = loess(male_recall ~ male_percent_predicted, span = loess.span)
  female_loess_fit = loess(female_recall ~ female_percent_predicted, span = loess.span)
  lines(male_percent_predicted[male_ordered], male_loess_fit$fitted[male_ordered], col=male_color, lwd=lwd)
  lines(female_percent_predicted[female_ordered], female_loess_fit$fitted[female_ordered], col=female_color, lwd=lwd)
  if (plot_points) {
    points(male_percent_predicted, male_recall,
           cex = cex.pt,
           pch = pch,  
           col = male_color)
    points(female_percent_predicted, female_recall,
           cex = cex.pt,
           pch = pch,  
           col = female_color)
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

  if (draw_legend) {
    points = c()
    if (plot_points) {
      points = c(pch, pch)
    }
    legend(x = legend_position,
           legend=legends,
           col=c(male_color, female_color),
           pch=points,
           lty = 1,
           lwd = lwd,
           cex=cex.legend,
           pt.cex=cex.pt,
           bty="n")
  }

  box(which="plot", bty="o", lwd = axis.lwd)
  dev.off()
}  

#################################
  
eu_file_suffixes = c("trs_5000.tes_15000",
                     "trs_10000.tes_15000",
                     "trs_20000.tes_10000",
                     "trs_-1.tes_10000")
method = "svm_k"

for (file_suffix in eu_file_suffixes) {
  setwd(dir = "~/research/gender_prediction/results/extended_indicators/subset_stats/stats-per-gender/")
  eu_stats_per_gender_file = paste("eu-stats-per-gender.", file_suffix, ".csv", sep="")
  eu_stats_per_gender_data = read.csv(file = eu_stats_per_gender_file, header = TRUE, sep = ",")
  
  # set the dir for output plots
  setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators/subset-gender-precision/")
  output_filename = paste("eu-precision-per-gender.", method, ".", file_suffix, ".pdf", sep="")
  generate_subset_precision_plots(filename = output_filename,
                                  data = eu_stats_per_gender_data,
                                  method = method,
                                  plot_margins = c(7,6.2,2.5,2.5),
                                  title = "",
                                  xlab = "Most Confident",
                                  ylab = "Precision (%)",
                                  x_min = 25,
                                  x_max = 100,
                                  y_min = 50,
                                  y_max = 100,
                                  plot_points = FALSE,
                                  pch = 19,
                                  lwd = 18,
                                  loess.span = 0.1,
                                  xticks = c(25,50,75,100),
                                  xlabels = c("25", "50", "75", "100"),
                                  x.padj = 0.6,
                                  xlab.margin = 4.5,
                                  yticks = c(50,60,70,80,90,100),
                                  ylabels = c("50","60","70","80","90","100"),
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
  
  setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators/subset-gender-recall/")
  output_filename = paste("eu-recall-per-gender.", method, ".", file_suffix, ".pdf", sep="")
  generate_subset_recall_plots(filename = output_filename,
                               data = eu_stats_per_gender_data,
                               method = method,
                               plot_margins = c(7,6.2,2.5,2.5),
                               title = "",
                               xlab = "Most Confident",
                               ylab = "Recall (%)",
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
  setwd(dir = "~/research/gender_prediction/results/extended_indicators/subset_stats/stats-per-gender/")
  sa_stats_per_gender_file = paste("sa-stats-per-gender.", file_suffix, ".csv", sep="")
  sa_stats_per_gender_data = read.csv(file = sa_stats_per_gender_file, header = TRUE, sep = ",")
  
  # set the dir for output plots
  setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators/subset-gender-precision/")
  output_filename = paste("sa-precision-per-gender.", method, ".", file_suffix, ".pdf", sep="")
  generate_subset_precision_plots(filename = output_filename,
                                  data = sa_stats_per_gender_data,
                                  method = method,
                                  plot_margins = c(7,6.2,2.5,2.5),
                                  title = "",
                                  xlab = "Most Confident",
                                  ylab = "Precision (%)",
                                  x_min = 25,
                                  x_max = 100,
                                  y_min = 50,
                                  y_max = 100,
                                  plot_points = FALSE,
                                  pch = 19,
                                  lwd = 18,
                                  loess.span = 0.1,
                                  xticks = c(25,50,75,100),
                                  xlabels = c("25", "50", "75", "100"),
                                  x.padj = 0.6,
                                  xlab.margin = 4.5,
                                  yticks = c(50,60,70,80,90,100),
                                  ylabels = c("50","60","70","80","90","100"),
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
  
  setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators/subset-gender-recall/")
  output_filename = paste("sa-recall-per-gender.", method, ".", file_suffix, ".pdf", sep="")
  generate_subset_recall_plots(filename = output_filename,
                               data = sa_stats_per_gender_data,
                               method = method,
                               plot_margins = c(7,6.2,2.5,2.5),
                               title = "",
                               xlab = "Most Confident",
                               ylab = "Recall (%)",
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