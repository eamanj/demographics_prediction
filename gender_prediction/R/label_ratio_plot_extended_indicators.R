# raw colors without alpha
svm_l_color="#E5DD3B"
svm_k_color="#B01E1B"
rf_color="#32567F"
logistic_color="#59FFB1"
knn_color="#C07737"

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

# the transformation takes FPR to 0 and takes TPR to 1. so the transformation of predicted ratio is
# calibrated_ratio = 1/(TPR-FPR) * predicted_ratio + (FPR)/(TPR-FPR)
add_calibrated_ratios = function(data, tpr, fpr) {
  tpr = data$train_true_female/data$train_female_size
  fpr = data$train_false_female/data$train_female_size
  predicted_ratios = data$test_predicted_ratio
  data$test_predicted_ratio_calibrated = (predicted_ratios/(tpr - fpr)) - (fpr/(tpr - fpr))
  return(data)
}

get_r_squared = function(data) {
  ss_res = sum((data$test_actual_ratio - data$test_predicted_ratio_calibrated)^2)
  ss_tot = sum((data$test_actual_ratio - mean(data$test_actual_ratio)^2))
  r_squared = 1 - (ss_res/ss_tot)
  return(r_squared)
}

generate_label_ratio_plot = function(filename,
                                     svm_l_data,
                                     svm_k_data,
                                     rf_data,
                                     logistic_data,
                                     knn_data,
                                     svm_l,
                                     svm_k,
                                     rf,
                                     logistic,
                                     knn,
                                     plot_margins,
                                     title,
                                     xlab,
                                     ylab,
                                     plot_points,
                                     pch,
                                     lwd,
                                     loess.span,
                                     ticks,
                                     labels,
                                     x.padj,
                                     xlab.margin,
                                     y.padj,
                                     ylab.margin,
                                     axis.lwd,
                                     draw_legend,
                                     legend_position,
                                     cex.pt,
                                     cex.axis,
                                     cex.main,
                                     cex.lab,
                                     cex.legend,
                                     font.main) {
  pdf(filename, width=20, height=20, pointsize = 25)
  par(mar = plot_margins)

  legends = character(0)
  legend_colors = character(0)

  plot(1,
       type="n",
       main = "",
       xlab="",
       ylab="",
       xlim=c(0, 1),
       ylim=c(0, 1),
       xaxt = "n",
       yaxt = "n",
       xaxs = "i",
       yaxs = "i",
       font.main = font.main)
  
  if (svm_l) {
    x = svm_l_data$test_actual_ratio
    y = svm_l_data$test_predicted_ratio_calibrated
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=svm_l_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = svm_l_color)
    }
    legends = append(legends, c("SVM-Linear"))
    legend_colors = append(legend_colors, c(svm_l_color))
  }
  if (svm_k) {
    x = svm_k_data$test_actual_ratio
    y = svm_k_data$test_predicted_ratio_calibrated
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=svm_k_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = svm_k_color)
    }
    legends = append(legends, c("SVM-RBF"))
    legend_colors = append(legend_colors, c(svm_k_color))
  }
  if (rf) {
    x = rf_data$test_actual_ratio
    y = rf_data$test_predicted_ratio_calibrated
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=rf_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = rf_color)
    }
    legends = append(legends, c("RF"))
    legend_colors = append(legend_colors, c(rf_color))
  }
  if (logistic) {
    x = logistic_data$test_actual_ratio
    y = logistic_data$test_predicted_ratio_calibrated
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=logistic_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = logistic_color)
    }
    legends = append(legends, c("Logistic"))
    legend_colors = append(legend_colors, c(logistic_color))
  }
  if (knn) {
    x = knn_data$test_actual_ratio
    y = knn_data$test_predicted_ratio_calibrated
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=knn_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = knn_color)
    }
    legends = append(legends, c("KNN"))
    legend_colors = append(legend_colors, c(knn_color))
  }

  segments(x0 = 0, y0 = 0, x1 = 1, y1 = 1, col = 'black', lty = 2, lwd = lwd)
    
  axis(side = 1, lwd = axis.lwd,
       at = ticks, labels = labels,
       cex.axis = cex.axis, cex.lab = cex.lab, padj=x.padj)
  axis(side = 2, lwd = axis.lwd,
       at = ticks, labels = labels,
       cex.axis = cex.axis, cex.lab = cex.lab, padj=y.padj)

  title(main = title, cex.main = cex.main)
  title(xlab = xlab, cex.lab = cex.lab, mgp = c(xlab.margin,1,0))
  title(ylab = ylab, cex.lab = cex.lab, mgp = c(ylab.margin,1,0))

  box(which="plot", bty="o", lwd = axis.lwd)

  if (draw_legend) {
    points = c()
    if (plot_points) {
      points = c(pch, pch, pch, pch, pch)
    }
    legend(x = legend_position,
           legend=legends,
           col=legend_colors,
           pch = points,
           lty = 1,
           lwd = lwd,
           cex=cex.legend,
           pt.cex=cex.pt,
           bty="n")
  }
  dev.off()
}


############################################

setwd(dir = "~/research/gender_prediction/results/extended_indicators/label-ratio/south_asian")
sa_svm_l_data = read.csv(file = "linear-svm/cleaned_south_asian_data_a2_c2_allweek_allday.csv.trs_10000.ts_5000.nt_8.sfs",
                         header = TRUE, sep = ",")
sa_svm_k_data = read.csv(file = "kernel-svm/cleaned_south_asian_data_a2_c2_allweek_allday.csv.trs_10000.ts_5000.nt_8.sfs",
                         header = TRUE, sep = ",")
sa_rf_data = read.csv(file = "random-forest/cleaned_south_asian_data_a2_c2_allweek_allday.csv.trs_10000.ts_5000.nt_8.sfs",
                      header = TRUE, sep = ",")
sa_logistic_data = read.csv(file = "logistic/cleaned_south_asian_data_a2_c2_allweek_allday.csv.trs_10000.ts_5000.nt_8.sfs",
                            header = TRUE, sep = ",")
sa_knn_data = read.csv(file = "knn/cleaned_south_asian_data_a2_c2_allweek_allday.csv.trs_10000.ts_5000.nt_8.sfs",
                       header = TRUE, sep = ",")

setwd(dir = "~/research/gender_prediction/results/extended_indicators/label-ratio/european/")
eu_svm_l_data = read.csv(file = "linear-svm/cleaned_european_data_a2_c2_allweek_allday.csv.10%sample.trs_10000.ts_10000.nt_8.sfs",
                         header = TRUE, sep = ",")
eu_svm_k_data = read.csv(file = "kernel-svm/cleaned_european_data_a2_c2_allweek_allday.csv.10%sample.trs_10000.ts_10000.nt_8.sfs",
                         header = TRUE, sep = ",")
eu_rf_data = read.csv(file = "random-forest/cleaned_european_data_a2_c2_allweek_allday.csv.10%sample.trs_10000.ts_10000.nt_8.sfs",
                      header = TRUE, sep = ",")
eu_logistic_data = read.csv(file = "logistic/cleaned_european_data_a2_c2_allweek_allday.csv.10%sample.trs_10000.ts_10000.nt_8.sfs",
                            header = TRUE, sep = ",")
eu_knn_data = read.csv(file = "knn/cleaned_european_data_a2_c2_allweek_allday.csv.10%sample.trs_10000.ts_10000.nt_8.sfs",
                       header = TRUE, sep = ",")

# add calibrated ratio column to each data set
sa_svm_l_data = add_calibrated_ratios(sa_svm_l_data)
sa_svm_k_data = add_calibrated_ratios(sa_svm_k_data)
sa_rf_data = add_calibrated_ratios(sa_rf_data)
sa_logistic_data = add_calibrated_ratios(sa_logistic_data)
sa_knn_data = add_calibrated_ratios(sa_knn_data)

eu_svm_l_data = add_calibrated_ratios(eu_svm_l_data)
eu_svm_k_data = add_calibrated_ratios(eu_svm_k_data)
eu_rf_data = add_calibrated_ratios(eu_rf_data)
eu_logistic_data = add_calibrated_ratios(eu_logistic_data)
eu_knn_data = add_calibrated_ratios(eu_knn_data)

setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators/")
generate_label_ratio_plot(filename = "sa_varying_label_ratio.pdf",
                          svm_l_data = sa_svm_l_data,
                          svm_k_data = sa_svm_k_data,
                          rf_data = sa_rf_data,
                          logistic_data = sa_logistic_data,
                          knn_data = sa_knn_data,
                          svm_l = TRUE,
                          svm_k = TRUE,
                          rf = TRUE,
                          logistic = TRUE,
                          knn = TRUE,
                          plot_margins = c(7,7,2,2),
                          title = "",
                          xlab = "Percentage of Women (True)",
                          ylab = "Percentage of Women (Predicted)",
                          plot_points = FALSE,
                          pch = 19,
                          lwd = 11,
                          loess.span = 0.3,
                          ticks = c(0,0.2,0.4,0.6,0.8,1),
                          labels = c("0", "0.2", "0.4", "0.6", "0.8", "1"),
                          x.padj = 0.5,
                          xlab.margin = 4.5,
                          y.padj = 0,
                          ylab.margin = 3.5,
                          axis.lwd = 11,
                          draw_legend = TRUE,
                          legend_position = "bottomright",
                          cex.pt = 2.5,
                          cex.axis = 3.3,
                          cex.main = 2.5,
                          cex.lab = 3,
                          cex.legend = 2.5,
                          font.main = 1)

generate_label_ratio_plot(filename = "eu_varying_label_ratio.pdf",
                          svm_l_data = eu_svm_l_data,
                          svm_k_data = eu_svm_k_data,
                          rf_data = eu_rf_data,
                          logistic_data = eu_logistic_data,
                          knn_data = eu_knn_data,
                          svm_l = TRUE,
                          svm_k = TRUE,
                          rf = TRUE,
                          logistic = TRUE,
                          knn = TRUE,
                          plot_margins = c(7,7,2,2),
                          title = "",
                          xlab = "Percentage of Women (True)",
                          ylab = "Percentage of Women (Predicted)",
                          plot_points = FALSE,
                          pch = 19,
                          lwd = 11,
                          loess.span = 0.3,
                          ticks = c(0,0.2,0.4,0.6,0.8,1),
                          labels = c("0", "0.2", "0.4", "0.6", "0.8", "1"),
                          x.padj = 0.5,
                          xlab.margin = 4.5,
                          y.padj = 0,
                          ylab.margin = 3.5,
                          axis.lwd = 11,
                          draw_legend = TRUE,
                          legend_position = "bottomright",
                          cex.pt = 2.5,
                          cex.axis = 3.3,
                          cex.main = 2.5,
                          cex.lab = 3,
                          cex.legend = 2.5,
                          font.main = 1)


# print mean abs error
mean(abs(sa_svm_l_data$test_actual_ratio - sa_svm_l_data$test_predicted_ratio_calibrated))
mean(abs(sa_svm_k_data$test_actual_ratio - sa_svm_k_data$test_predicted_ratio_calibrated))
mean(abs(sa_rf_data$test_actual_ratio - sa_rf_data$test_predicted_ratio_calibrated))
mean(abs(sa_logistic_data$test_actual_ratio - sa_logistic_data$test_predicted_ratio_calibrated))
mean(abs(sa_knn_data$test_actual_ratio - sa_knn_data$test_predicted_ratio_calibrated))

mean(abs(eu_svm_l_data$test_actual_ratio - eu_svm_l_data$test_predicted_ratio_calibrated))
mean(abs(eu_svm_k_data$test_actual_ratio - eu_svm_k_data$test_predicted_ratio_calibrated))
mean(abs(eu_rf_data$test_actual_ratio - eu_rf_data$test_predicted_ratio_calibrated))
mean(abs(eu_logistic_data$test_actual_ratio - eu_logistic_data$test_predicted_ratio_calibrated))
mean(abs(eu_knn_data$test_actual_ratio - eu_knn_data$test_predicted_ratio_calibrated))

# Get r-squared
get_r_squared(sa_svm_l_data)
get_r_squared(sa_svm_k_data)
get_r_squared(sa_rf_data)
get_r_squared(sa_logistic_data)
get_r_squared(sa_knn_data)

get_r_squared(eu_svm_l_data)
get_r_squared(eu_svm_k_data)
get_r_squared(eu_rf_data)
get_r_squared(eu_logistic_data)
get_r_squared(eu_knn_data)
