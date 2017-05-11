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


generate_train_size_plot = function(filename,
                                    size,
                                    svm_l_data,
                                    svm_k_data,
                                    rf_data,
                                    logistic_data,
                                    knn_data,
                                    largest_stats,
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
                                    xdashline,
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
  pdf(filename, width=size[1], height=size[2], pointsize = 25)
  par(mar = plot_margins)

  legends = character(0)
  legend_colors = character(0)

  # extract the largest point info
  fake_largest_train_size = NULL
  largest_svm_l_accuracy = NULL
  largest_svm_k_accuracy = NULL
  largest_rf_accuracy = NULL
  largest_logistic_accuracy = NULL
  largest_knn_accuracy = NULL
  if (!is.null(largest_stats)) {
    fake_largest_train_size = largest_stats[1]
    largest_svm_l_accuracy = largest_stats[2]
    largest_svm_k_accuracy = largest_stats[3]
    largest_rf_accuracy = largest_stats[4]
    largest_logistic_accuracy = largest_stats[5]
    largest_knn_accuracy = largest_stats[6]
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
  
  if (svm_l) {
    x = svm_l_data$train_size
    y = svm_l_data[[column]]
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=svm_l_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = svm_l_color)
    }
    if (!is.null(fake_largest_train_size)) {
      points(fake_largest_train_size, largest_svm_l_accuracy, cex = cex.pt, pch = pch, col = svm_l_color)
    }
    legends = append(legends, c("SVM-Linear"))
    legend_colors = append(legend_colors, c(svm_l_color))
  }

  if (svm_k) {
    x = svm_k_data$train_size
    y = svm_k_data[[column]]
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=svm_k_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = svm_k_color)
    }
    if (!is.null(fake_largest_train_size)) {
      points(fake_largest_train_size, largest_svm_k_accuracy, cex = cex.pt, pch = pch, col = svm_k_color)
    }
    legends = append(legends, c("SVM-RBF"))
    legend_colors = append(legend_colors, c(svm_k_color))
  }

  if (rf) {
    x = rf_data$train_size
    y = rf_data[[column]]
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=rf_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = rf_color)
    }
    if (!is.null(fake_largest_train_size)) {
      points(fake_largest_train_size, largest_rf_accuracy, cex = cex.pt, pch = pch, col = rf_color)
    }
    legends = append(legends, c("RF"))
    legend_colors = append(legend_colors, c(rf_color))
  }
  
  if (logistic) {
    x = logistic_data$train_size
    y = logistic_data[[column]]
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=logistic_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = logistic_color)
    }
    if (!is.null(fake_largest_train_size)) {
      points(fake_largest_train_size, largest_logistic_accuracy, cex = cex.pt, pch = pch, col = logistic_color)
    }
    legends = append(legends, c("Logistic"))
    legend_colors = append(legend_colors, c(logistic_color))
  }
  
  if (knn) {
    x = knn_data$train_size
    y = knn_data[[column]]
    loess_fit = loess(y ~ x, span = loess.span)
    ordered = order(x)
    lines(x[ordered], loess_fit$fitted[ordered], col=knn_color, lwd=lwd)
    if (plot_points) {
      points(x, y,
             cex = cex.pt,
             pch = pch,  
             col = knn_color)
    }
    if (!is.null(fake_largest_train_size)) {
      points(fake_largest_train_size, largest_knn_accuracy, cex = cex.pt, pch = pch, col = knn_color)
    }
    legends = append(legends, c("KNN"))
    legend_colors = append(legend_colors, c(knn_color))
  }
  if (xdashline != 0) {
    abline(v = xdashline, col = 'black', lty = 2, lwd = axis.lwd)
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
    legend(x = legend_position[1],
           y = legend_position[2],
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

############################################

setwd(dir = "~/research/gender_prediction/results/extended_indicators/train-size/south_asian")
sa_svm_l_data = read.csv(file = "linear-svm/cleaned_south_asian_data_a2_c2_top200.csv.ts_10000.nt_7.sc_10.sk_linear",
                         header = TRUE, sep = ",")
sa_svm_k_data = read.csv(file = "kernel-svm/cleaned_south_asian_data_a2_c2_top200.csv.ts_7000.nt_7.sc_100.sk_rbf.sg_0.001",
                         header = TRUE, sep = ",")
sa_rf_data = read.csv(file = "random-forest/cleaned_south_asian_data_a2_c2_top200.csv.ts_10000.nt_7.rnt_800.rc_entropy.rmf_0.7.rmss_20.rmsl_10",
                      header = TRUE, sep = ",")
sa_logistic_data = read.csv(file = "logistic/cleaned_south_asian_data_a2_c2_top200.csv.ts_10000.nt_7.lop_l1.loc_1",
                            header = TRUE, sep = ",")
sa_knn_data = read.csv(file = "knn/cleaned_south_asian_data_a2_c2_top200.csv.ts_10000.nt_7.knn_150.kw_distance.ka_ball_tree.km_manhattan",
                       header = TRUE, sep = ",")

setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators/")
generate_train_size_plot(filename = "sa_varying_train_size.pdf",
                         size=c(20,13),
                         svm_l_data = sa_svm_l_data,
                         svm_k_data = sa_svm_k_data,
                         rf_data = sa_rf_data,
                         logistic_data = sa_logistic_data,
                         knn_data = sa_knn_data,
                         largest_stats = NULL,
                         svm_l = TRUE,
                         svm_k = TRUE,
                         rf = TRUE,
                         logistic = TRUE,
                         knn = TRUE,
                         column = "test_accuracy",
                         plot_margins = c(3.8,4,2,10.7),
                         title = "",
                         xlab = "Training Size",
                         ylab = "Accuracy (%)",
                         x_min = 500,
                         x_max = 15000,
                         y_min = 60,
                         y_max = 80,
                         plot_points = FALSE,
                         pch = 19,
                         lwd = 10,
                         loess.span = 0.3,
                         xdashline = 10000,
                         xticks = c(0,5000,10000,15000),
                         xlabels = c("0","5K","10K","15K"),
                         x.padj = 0,
                         xlab.margin = 2.4,
                         yticks = c(60,65,70,75,80),
                         ylabels = c("60", "65", "70", "75", "80"),
                         y.padj = 0,
                         ylab.margin = 2.5,
                         axis.lwd = 10,
                         draw_legend = TRUE,
                         legend_position = c(9700, 66.8),
                         font.main = 1,
                         cex.pt = 1.8,
                         cex.axis = 1.8,
                         cex.main = 2.5,
                         cex.lab = 1.8,
                         cex.legend = 1.5)

setwd(dir = "~/research/gender_prediction/results/extended_indicators/train-size/european")
eu_svm_l_data = read.csv(file = "linear-svm/cleaned_european_data_a2_c2_top200.csv.20%sample.ts_10000.nt_5.sc_1.sk_linear",
                         header = TRUE, sep = ",")
eu_svm_k_data = read.csv(file = "kernel-svm/cleaned_european_data_a2_c2_top200.csv.20%sample.ts_10000.nt_5.sc_1000.sk_rbf.sg_0.001",
                         header = TRUE, sep = ",")
eu_rf_data = read.csv(file = "random-forest/cleaned_european_data_a2_c2_top200.csv.20%sample.ts_10000.nt_5.rnt_400.rc_entropy.rmf_sqrt.rmss_5.rmsl_2",
                      header = TRUE, sep = ",")
eu_logistic_data = read.csv(file = "logistic/cleaned_european_data_a2_c2_top200.csv.20%sample.ts_10000.nt_5.lop_l1.loc_1",
                            header = TRUE, sep = ",")
eu_knn_data = read.csv(file = "knn/cleaned_european_data_a2_c2_top200.csv.20%sample.ts_10000.nt_5.knn_70.kw_distance.ka_ball_tree.km_manhattan",
                       header = TRUE, sep = ",")

# delete rows after 15000
eu_svm_l_data = eu_svm_l_data[eu_svm_l_data$train_size<15001,]
eu_svm_k_data = eu_svm_k_data[eu_svm_k_data$train_size<15001,]
eu_rf_data = eu_rf_data[eu_rf_data$train_size<15001,]
eu_logistic_data = eu_logistic_data[eu_logistic_data$train_size<15001,]
eu_knn_data = eu_knn_data[eu_knn_data$train_size<15001,]

# largest train size and its accuracy to be plotted as a single point
eu_fake_largest_train_size = 17000
eu_largest_svm_l = 75.04
eu_largest_svm_k = 75.36
eu_largest_rf = 74.55
eu_largest_logistic = 74.90
eu_largest_knn = 70.62
setwd(dir = "~/research/gender_prediction/plots/results/extended_indicators/")
generate_train_size_plot(filename = "eu_varying_train_size.pdf",
                         size=c(20,13),
                         svm_l_data = eu_svm_l_data,
                         svm_k_data = eu_svm_k_data,
                         rf_data = eu_rf_data,
                         logistic_data = eu_logistic_data,
                         knn_data = eu_knn_data,
                         largest_stats = c(eu_fake_largest_train_size, eu_largest_svm_l, eu_largest_svm_k, eu_largest_rf, eu_largest_logistic, eu_largest_knn),
                         svm_l = TRUE,
                         svm_k = TRUE,
                         rf = TRUE,
                         logistic = TRUE,
                         knn = TRUE,
                         column = "test_accuracy",
                         plot_margins = c(3.8,4,2,2.9),
                         title = "",
                         xlab = "Training Size",
                         ylab = "Accuracy (%)",
                         x_min = 500,
                         x_max = eu_fake_largest_train_size+1500,
                         y_min = 60,
                         y_max = 80,
                         plot_points = FALSE,
                         pch = 19,
                         lwd = 10,
                         loess.span = 0.3,
                         xdashline = 10000,
                         xticks = c(0,5000,10000,15000,eu_fake_largest_train_size),
                         xlabels = c("0","5K","10K","15K","500K"),
                         x.padj = 0,
                         xlab.margin = 2.4,
                         yticks = c(60,65,70,75,80),
                         ylabels = c("60", "65", "70", "75", "80"),
                         y.padj = 0,
                         ylab.margin = 2.5,
                         axis.lwd = 10,
                         draw_legend = TRUE,
                         legend_position = c(9700, 66.8),
                         font.main = 1,
                         cex.pt = 1.8,
                         cex.axis = 1.8,
                         cex.main = 2.5,
                         cex.lab = 1.8,
                         cex.legend = 1.5)
