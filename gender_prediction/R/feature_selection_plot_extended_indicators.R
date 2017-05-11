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

generate_feature_selection_plot = function(filename,
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
                                           column,
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
  
  if (svm_l) {
    x = svm_l_data$num_features_selected
    y = svm_l_data[[column]]
    x_ext = c(0,x)
    y_ext = c(min(y), y)
    loess_fit = loess(y_ext ~ x_ext, span = loess.span)
    ordered = order(x_ext)
    lines(x_ext[ordered], loess_fit$fitted[ordered], col=svm_l_color, lwd=lwd)
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
    x = svm_k_data$num_features_selected
    y = svm_k_data[[column]]
    x_ext =c(0,x)
    y_ext = c(min(y), y)
    loess_fit = loess(y_ext ~ x_ext, span = loess.span)
    ordered = order(x_ext)
    lines(x_ext[ordered], loess_fit$fitted[ordered], col=svm_k_color, lwd=lwd)
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
    x = rf_data$num_features_selected
    y = rf_data[[column]]
    x_ext = c(0,x)
    y_ext = c(min(y), y)
    loess_fit = loess(y_ext ~ x_ext, span = loess.span)
    ordered = order(x_ext)
    lines(x_ext[ordered], loess_fit$fitted[ordered], col=rf_color, lwd=lwd)
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
    x = logistic_data$num_features_selected
    y = logistic_data[[column]]
    x_ext = c(0,x)
    y_ext = c(min(y), y)
    loess_fit = loess(y_ext ~ x_ext, span = loess.span)
    ordered = order(x_ext)
    lines(x_ext[ordered], loess_fit$fitted[ordered], col=logistic_color, lwd=lwd)
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
    x = knn_data$num_features_selected
    y = knn_data[[column]]
    x_ext = c(0,x)
    y_ext = c(min(y), y)
    loess_fit = loess(y_ext ~ x_ext, span = loess.span)
    ordered = order(x_ext)
    lines(x_ext[ordered], loess_fit$fitted[ordered], col=knn_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
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


#######################################################

setwd(dir = "~/research/gender_prediction/results/extended_indicators/feature-selection-l1-svm/south_asian")
sa_svm_l_data = read.csv(file = "linear-svm/cleaned_south_asian_data_a2_c2_top200.csv.ts_10000.c_10.k_linear",
                         header = TRUE, sep = ",")
sa_svm_k_data = read.csv(file = "kernel-svm/cleaned_south_asian_data_a2_c2_top200.csv.ts_10000.c_1000.k_rbf.g_0.001",
                         header = TRUE, sep = ",")
sa_rf_data = read.csv(file = "random-forest/cleaned_south_asian_data_a2_c2_top200.csv.ts_10000.nt_400.cr_entropy.mf_0.7.mss_20.msl_10",
                      header = TRUE, sep = ",")
sa_logistic_data = read.csv(file = "logistic/cleaned_south_asian_data_a2_c2_top200.csv.ts_10000.lp_l1.lco_1",
                            header = TRUE, sep = ",")
sa_knn_data = read.csv(file = "knn/cleaned_south_asian_data_a2_c2_top200.csv.ts_10000.nn_150.w_distance.a_ball_tree.m_manhattan",
                       header = TRUE, sep = ",")

setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators")
generate_feature_selection_plot(filename = "sa_feature_selection.pdf",
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
                                column = "avg_accuracy",
                                plot_margins = c(7,6,2,2.5),
                                title = "",
                                xlab = "Top Features",
                                ylab = "Accuracy (%)",
                                x_min = 0,
                                x_max = 100,
                                y_min = 60,
                                y_max = 80,
                                plot_points = FALSE,
                                pch = 19,
                                lwd = 18,
                                loess.span = 0.3,
                                xticks = c(0,25,50,75,100),
                                xlabels = c("0", "25", "50", "75", "100"),
                                x.padj = 0.5,
                                xlab.margin = 4.5,
                                yticks = c(60,65,70,75,80),
                                ylabels = c("60", "65", "70", "75", "80"),
                                y.padj = 0,
                                ylab.margin = 3.5,
                                axis.lwd = 11,
                                draw_legend = TRUE,
                                legend_position = "bottomright",
                                font.main = 1,
                                cex.pt = 1,
                                cex.axis = 3.3,
                                cex.main = 2.5,
                                cex.lab = 3,
                                cex.legend = 2.5)

setwd(dir = "~/research/gender_prediction/results/extended_indicators/feature-selection-l1-svm/european")
eu_svm_l_data = read.csv(file = "linear-svm/cleaned_european_data_a2_c2_top200.csv.15%sample.ts_15000.c_1.k_linear",
                         header = TRUE, sep = ",")
eu_svm_k_data = read.csv(file = "kernel-svm/cleaned_european_data_a2_c2_top200.csv.15%sample.ts_25000.c_1000.k_rbf.g_0.001",
                         header = TRUE, sep = ",")
eu_rf_data = read.csv(file = "random-forest/cleaned_european_data_a2_c2_top200.csv.15%sample.ts_25000.nt_400.cr_entropy.mf_sqrt.mss_5.msl_2",
                      header = TRUE, sep = ",")
eu_logistic_data = read.csv(file = "logistic/cleaned_european_data_a2_c2_top200.csv.15%sample.ts_25000.lp_l1.lco_1",
                            header = TRUE, sep = ",")
eu_knn_data = read.csv(file = "knn/cleaned_european_data_a2_c2_top200.csv.15%sample.ts_25000.nn_70.w_distance.a_ball_tree.m_manhattan",
                       header = TRUE, sep = ",")

setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators")
generate_feature_selection_plot(filename = "eu_feature_selection.pdf",
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
                                column = "avg_accuracy",
                                plot_margins = c(7,6,2,2.5),
                                title = "",
                                xlab = "Top Features",
                                ylab = "Accuracy (%)",
                                x_min = 0,
                                x_max = 100,
                                y_min = 60,
                                y_max = 80,
                                plot_points = FALSE,
                                pch = 19,
                                lwd = 18,
                                loess.span = 0.3,
                                xticks = c(0,25,50,75,100),
                                xlabels = c("0", "25", "50", "75", "100"),
                                x.padj = 0.5,
                                xlab.margin = 4.5,
                                yticks = c(60,65,70,75,80),
                                ylabels = c("60", "65", "70", "75", "80"),
                                y.padj = 0,
                                ylab.margin = 3.5,
                                axis.lwd = 11,
                                draw_legend = TRUE,
                                legend_position = "bottomright",
                                font.main = 1,
                                cex.pt = 1,
                                cex.axis = 3.3,
                                cex.main = 2.5,
                                cex.lab = 3,
                                cex.legend = 2.5)
