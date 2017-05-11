# raw colors without alpha
accuracy_col = "dodgerblue2"
accuracy_alpha = 0.9
auc_col = "indianred"
auc_alpha = 1
coverage_col = "forestgreen"
coverage_alpha = 0.9

# add alpha channel to colors
accuracy_col = col2rgb(accuracy_col)/255
accuracy_col = rgb(red = accuracy_col[1], green = accuracy_col[2], blue = accuracy_col[3], alpha = accuracy_alpha)

auc_col = col2rgb(auc_col)/255
auc_col = rgb(red = auc_col[1], green = auc_col[2], blue = auc_col[3], alpha = auc_alpha)

coverage_col = col2rgb(coverage_col)/255
coverage_col = rgb(red = coverage_col[1], green = coverage_col[2], blue = coverage_col[3], alpha = coverage_alpha)

generate_data_filtering_plot = function(filename,
                                        data,
                                        colors,
                                        plot_margins,
                                        title,
                                        xlab,
                                        ylab1,
                                        ylab2,
                                        x_min,
                                        x_max,
                                        y_min1,
                                        y_max1,
                                        y_min2,
                                        y_max2,
                                        pch1,
                                        pch2,
                                        pch3,
                                        lty1,
                                        lty2,
                                        lty3,
                                        lwd,
                                        xticks,
                                        xlabels,
                                        xtitle.margin,
                                        xlab.margin,
                                        yticks1,
                                        ylabels1,
                                        yticks2,
                                        ylabels2,
                                        ytitle1.margin,
                                        ylab1.margin,
                                        ytitle2.margin,
                                        ylab2.margin,
                                        axis.lwd,
                                        draw_legend,
                                        legend_position,
                                        font.main,
                                        cex.pt,
                                        cex.axis,
                                        cex.main,
                                        cex.lab,
                                        cex.legend,
                                        seg.len) {
  pdf(filename, width=23, height=20, pointsize = 25)
  par(mar = plot_margins)

  x = data$min_active_days
  y1 = data$test_accuracy / 100.0
  plot(x, y1,
       axes = FALSE,
       main = "",
       xlab="",
       ylab="",
       xlim=c(x_min, x_max),
       ylim=c(y_min1, y_max1),
       xaxs = "i",
       yaxs = "i",
       type = "l",
       col=colors[1],
       lwd=lwd,
       lty=lty1,
       font.main = font.main)
  points(x, y1,
         cex = cex.pt,
         pch = pch1,  
         col = colors[1])
  
  y2 = data$test_AUC
  lines(x, y2, col=colors[2], lwd=lwd, lty=lty2)
  points(x, y2,
         cex = cex.pt,
         pch = pch2,  
         col = colors[2])
  axis(side = 1, lwd = axis.lwd,
       at = xticks, labels = xlabels,
       mgp = c(xtitle.margin,xlab.margin,0),
       cex.axis = cex.axis, cex.lab = cex.lab)
  axis(side = 2, lwd = axis.lwd,
       at = yticks1, labels = ylabels1,
       mgp = c(ytitle1.margin,ylab1.margin,0),
       cex.axis = cex.axis, cex.lab = cex.lab)
  par(new=TRUE)
  y3 = data$percentage_data
  plot(x, y3,
       axes = FALSE,
       main = "",
       xlab="",
       ylab="",
       xlim=c(x_min, x_max),
       ylim=c(y_min2, y_max2),
       xaxs = "i",
       yaxs = "i",
       type = "l",
       col=colors[3],
       lwd=lwd,
       lty=lty3,
       font.main = font.main)
  points(x, y3,
         cex = cex.pt,
         pch = pch3,  
         col = colors[3])
  axis(side = 4, lwd = axis.lwd,
       at = yticks2, labels = ylabels2,
       mgp = c(ytitle2.margin,ylab2.margin,0),
       cex.axis = cex.axis, cex.lab = cex.lab)

  box(which="plot", bty="o", lwd = axis.lwd)
  
  title(main = title, cex.main = cex.main, font = font.main)
  mtext(text = xlab,  side = 1, line = xtitle.margin, cex = cex.lab, font = font.main)
  mtext(text = ylab1, side = 2, line = ytitle1.margin, cex = cex.lab, font = font.main)
  mtext(text = ylab2, side = 4, line = ytitle2.margin, cex = cex.lab, font = font.main)
  
  if (draw_legend) {
    legends = c("Accuracy", "AUC", "Coverage")
    legend(x = legend_position,
           legend=legends,
           col=colors,
           pch=c(pch1, pch2, pch3),
           lty = c(lty1, lty2, lty3),
           seg.len = seg.len,
           lwd = lwd,
           cex=cex.legend,
           pt.cex=cex.pt,
           bty="n")
  }
  dev.off()
}


#######################################################

setwd(dir = "~/research/gender_prediction/results/extended_indicators/data-filtering/south_asian")
sa_svm_l_data = read.csv(file = "linear-svm/cleaned_south_asian_data_a0_c0.csv.trs_15000.tes_8000.nt_5",
                         header = TRUE, sep = ",")
sa_svm_k_data = read.csv(file = "kernel-svm/cleaned_south_asian_data_a0_c0.csv.trs_15000.tes_8000.nt_5",
                         header = TRUE, sep = ",")
sa_rf_data = read.csv(file = "random-forest/cleaned_south_asian_data_a0_c0.csv.trs_15000.tes_8000.nt_7",
                      header = TRUE, sep = ",")
sa_logistic_data = read.csv(file = "logistic/cleaned_south_asian_data_a0_c0.csv.trs_15000.tes_8000.nt_9",
                            header = TRUE, sep = ",")
sa_knn_data = read.csv(file = "knn/cleaned_south_asian_data_a0_c0.csv.trs_15000.tes_8000.nt_5",
                       header = TRUE, sep = ",")

setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators")
generate_data_filtering_plot(filename = "sa_data_filtering.pdf",
                             data = sa_svm_k_data,
                             colors=c(accuracy_col, auc_col, coverage_col),
                             plot_margins = c(6,6.5,2.3,6.5),
                             title = "",
                             xlab = "Min Active Days per Week",
                             ylab1 = "Accuracy/AUC",
                             ylab2 = "Coverage (%)",
                             x_min = 0,
                             x_max = 6.5,
                             y_min1 = 0.73,
                             y_max1 = 0.86,
                             y_min2 = 65,
                             y_max2 = 102,
                             pch1 = 19,
                             pch2 = 15,
                             pch3 = 17,
                             lty1 = 1,
                             lty2 = 2,
                             lty3 = 3,
                             lwd = 18,
                             xticks = c(0,1,2,3,4,5,6),
                             xlabels = c("0","1","2","3","4","5","6"),
                             xtitle.margin = 4,
                             xlab.margin = 1.9,
                             yticks1 = c(0.75, 0.78, 0.81, 0.84),
                             ylabels1 = c("0.75", "0.78", "0.81", "0.84"),
                             yticks2 = c(70, 80, 90, 100),
                             ylabels2 = c("70", "80", "90", "100"),
                             ytitle1.margin = 4,
                             ylab1.margin = 1.5,
                             ytitle2.margin = 4,
                             ylab2.margin = 2,
                             axis.lwd = 11,
                             draw_legend = TRUE,
                             legend_position = "topright",
                             font.main = 1,
                             cex.pt = 2.5,
                             cex.axis = 3.2,
                             cex.main = 2.5,
                             cex.lab = 2.9,
                             cex.legend = 2.5,
                             seg.len = 2.7)

setwd(dir = "~/research/gender_prediction/results/extended_indicators/data-filtering/european")
eu_svm_l_data = read.csv(file = "linear-svm/cleaned_european_data_a0_c0.csv.20%sample.trs_15000.tes_10000.nt_5",
                         header = TRUE, sep = ",")
eu_svm_k_data = read.csv(file = "kernel-svm/cleaned_european_data_a0_c0.csv.20%sample.trs_15000.tes_10000.nt_5",
                         header = TRUE, sep = ",")
eu_rf_data = read.csv(file = "random-forest/cleaned_european_data_a0_c0.csv.20%sample.trs_15000.tes_10000.nt_6",
                      header = TRUE, sep = ",")
eu_logistic_data = read.csv(file = "logistic/cleaned_european_data_a0_c0.csv.20%sample.trs_15000.tes_10000.nt_7",
                            header = TRUE, sep = ",")
eu_knn_data = read.csv(file = "knn/cleaned_european_data_a0_c0.csv.20%sample.trs_15000.tes_10000.nt_5",
                       header = TRUE, sep = ",")

setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators")
generate_data_filtering_plot(filename = "eu_data_filtering.pdf",
                             data = eu_svm_l_data,
                             colors=c(accuracy_col, auc_col, coverage_col),
                             plot_margins = c(6,6.5,2.3,6.5),
                             title = "",
                             xlab = "Min Active Days per Week",
                             ylab1 = "Accuracy/AUC",
                             ylab2 = "Coverage (%)",
                             x_min = 0,
                             x_max = 6.5,
                             y_min1 = 0.73,
                             y_max1 = 0.86,
                             y_min2 = 65,
                             y_max2 = 102,
                             pch1 = 19,
                             pch2 = 15,
                             pch3 = 17,
                             lty1 = 1,
                             lty2 = 2,
                             lty3 = 3,
                             lwd = 18,
                             xticks = c(0,1,2,3,4,5,6),
                             xlabels = c("0","1","2","3","4","5","6"),
                             xtitle.margin = 4,
                             xlab.margin = 1.9,
                             yticks1 = c(0.75, 0.78, 0.81, 0.84),
                             ylabels1 = c("0.75", "0.78", "0.81", "0.84"),
                             yticks2 = c(70, 80, 90, 100),
                             ylabels2 = c("70", "80", "90", "100"),
                             ytitle1.margin = 4,
                             ylab1.margin = 1.5,
                             ytitle2.margin = 4,
                             ylab2.margin = 2,
                             axis.lwd = 11,
                             draw_legend = TRUE,
                             legend_position = "topright",
                             font.main = 1,
                             cex.pt = 2.5,
                             cex.axis = 3.2,
                             cex.main = 2.5,
                             cex.lab = 2.9,
                             cex.legend = 2.5,
                             seg.len = 2.7)