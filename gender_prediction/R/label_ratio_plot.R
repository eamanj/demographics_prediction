
generate_label_ratio_plot = function(svm_l,
                                     svm_k,
                                     rf,
                                     logistic,
                                     knn,
                                     cex.axis,
                                     cex.lab,
                                     cex.main,
                                     cex) {
  filename="varying_label_ratio.png"
  png(filename, width=1900, height=2000, units="px", pointsize = 25)
  layout(matrix(c(1,2), 2, 1, byrow = TRUE))

  legends = character(0)
  legend_colors = character(0)
  par(mar = c(4.5, 4.8, 2.5, 1))
  title = paste("Train size: ", mean(sa_label_ratio_linear_svm_data$train_size))
  plot(1,
       type="n",
       main = title,
       xlab="Actual Female to Test Size Ratio",
       ylab="Predicted Female to Test Size Ratio",
       xlim=c(0, 1),
       ylim=c(0, 1),
       lab=c(10,10,10),
       cex.main=cex.main,
       cex.lab = cex.lab,
       cex.axis = cex.axis,
       bty="o")
  
  if (svm_l) {
    lines(sa_label_ratio_linear_svm_data$test_actual_ratio,
          sa_label_ratio_linear_svm_data$test_predicted_ratio,
          type = "p",
          cex = cex,
          pch = 19,  
          col ="indianred")
    legends = append(legends, c("SA Linear SVM"))
    legend_colors = append(legend_colors, c("indianred"))
  }
  if (svm_k) {
    lines(sa_label_ratio_kernel_svm_data$test_actual_ratio,
          sa_label_ratio_kernel_svm_data$test_predicted_ratio,
          type = "p",
          cex = cex,
          pch = 19,  
          col="forestgreen")
    legends = append(legends, c("SA Kernel SVM"))
    legend_colors = append(legend_colors, c("forestgreen"))
  }
  if (rf) {
    lines(sa_label_ratio_rf_data$test_actual_ratio,
          sa_label_ratio_rf_data$test_predicted_ratio,
          type = "p",
          cex = cex,
          pch = 19,  
          col="dodgerblue")
    legends = append(legends, c("SA RF"))
    legend_colors = append(legend_colors, c("dodgerblue"))
  }
  if (logistic) {
    lines(sa_label_ratio_logistic_data$test_actual_ratio,
          sa_label_ratio_logistic_data$test_predicted_ratio,
          type = "p",
          cex = cex,
          pch = 19,  
          col="khaki")
    legends = append(legends, c("SA Logistic"))
    legend_colors = append(legend_colors, c("khaki"))
  }
  if (knn) {
    lines(sa_label_ratio_knn_data$test_actual_ratio,
          sa_label_ratio_knn_data$test_predicted_ratio,
          type = "p",
          cex = cex,
          pch = 19,  
          col="slategray")
    legends = append(legends, c("SA KNN"))
    legend_colors = append(legend_colors, c("slategray"))
  }
  
  legend(x = "bottomright",
         legend=legends,
         col=legend_colors,
         pch=c(19,19,19,19,19,19),
         cex=cex,
         pt.cex=cex,
         bty="n")
  
  # plot european data now  
  legends = character(0)
  legend_colors = character(0)
  par(mar = c(4.5, 4.8, 2.5, 1))
  title = paste("Train size: ", mean(eu_label_ratio_linear_svm_data$train_size))
  plot(1,
       type="n",
       main = title,
       xlab="Actual Female to Test Size Ratio",
       ylab="Predicted Female to Test Size Ratio",
       xlim=c(0, 1),
       ylim=c(0, 1),
       lab=c(10,10,10),
       cex.main=cex.main,
       cex.lab = cex.lab,
       cex.axis = cex.axis,
       bty="o")
  
  if (svm_l) {
    points(eu_label_ratio_linear_svm_data$test_actual_ratio,
           eu_label_ratio_linear_svm_data$test_predicted_ratio,
           type = "p",
           cex = cex,
           pch = 19,  
           col ="indianred")
    legends = append(legends, c("EU Linear SVM"))
    legend_colors = append(legend_colors, c("indianred"))
  }
  if (svm_k) {
    points(eu_label_ratio_kernel_svm_data$test_actual_ratio,
           eu_label_ratio_kernel_svm_data$test_predicted_ratio,
           type = "p",
           cex = cex,
           pch = 19,  
           col="forestgreen")
    legends = append(legends, c("EU Kernel SVM"))
    legend_colors = append(legend_colors, c("forestgreen"))
  }
  if (rf) {
    points(eu_label_ratio_rf_data$test_actual_ratio,
           eu_label_ratio_rf_data$test_predicted_ratio,
           type = "p",
           cex = cex,
           pch = 19,  
           col="dodgerblue")
    legends = append(legends, c("EU RF"))
    legend_colors = append(legend_colors, c("dodgerblue"))
  }
  if (logistic) {
    lines(eu_label_ratio_logistic_data$test_actual_ratio,
          eu_label_ratio_logistic_data$test_predicted_ratio,
          type = "p",
          cex = cex,
          pch = 19,  
          col="khaki")
    legends = append(legends, c("EU Logistic"))
    legend_colors = append(legend_colors, c("khaki"))
  }
  if (knn) {
    lines(eu_label_ratio_knn_data$test_actual_ratio,
          eu_label_ratio_knn_data$test_predicted_ratio,
          type = "p",
          cex = cex,
          pch = 19,  
          col="slategray")
    legends = append(legends, c("EU KNN"))
    legend_colors = append(legend_colors, c("slategray"))
  }
  
  legend(x = "bottomright",
         legend=legends,
         col=legend_colors,
         pch=c(19,19,19,19,19,19),
         cex=cex,
         pt.cex=cex,
         bty="n")
  dev.off()
}

############################################

setwd(dir = "~/research/gender_prediction/results/label-ratio/south_asian")
sa_label_ratio_linear_svm_data = read.csv(file = "linear-svm/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.sc_10.sk_linear",
                                          header = TRUE, sep = ",")
sa_label_ratio_kernel_svm_data = read.csv(file = "kernel-svm/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.sc_1000.sk_rbf.sg_0.01",
                                          header = TRUE, sep = ",")
sa_label_ratio_rf_data = read.csv(file = "random-forest/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.rnt_400.rc_entropy.rmf_0.5.rmss_10.rmsl_5",
                                  header = TRUE, sep = ",")
sa_label_ratio_logistic_data = read.csv(file = "logistic/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.lop_l1.loc_10",
                                        header = TRUE, sep = ",")
sa_label_ratio_knn_data = read.csv(file = "knn/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.knn_150.kw_distance.ka_ball_tree.km_manhattan",
                                   header = TRUE, sep = ",")

setwd(dir = "~/research/gender_prediction/results/label-ratio/european/")
# TODO: FIX ONCE EU IS READY
setwd(dir = "~/research/gender_prediction/results/label-ratio/south_asian")
eu_label_ratio_linear_svm_data = read.csv(file = "linear-svm/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.sc_10.sk_linear.sb",
                                          header = TRUE, sep = ",")
eu_label_ratio_kernel_svm_data = read.csv(file = "kernel-svm/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.sc_1000.sk_rbf.sg_0.01.sb",
                                          header = TRUE, sep = ",")
eu_label_ratio_rf_data = read.csv(file = "random-forest/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.rnt_400.rc_entropy.rmf_0.5.rmss_10.rmsl_5.sb",
                                  header = TRUE, sep = ",")
eu_label_ratio_logistic_data = read.csv(file = "logistic/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.lop_l1.loc_10.sb",
                                        header = TRUE, sep = ",")
eu_label_ratio_knn_data = read.csv(file = "knn/cleaned_south_asian_data_a2_c2.csv.trs_1900.ts_2000.nt_10.knn_150.kw_distance.ka_ball_tree.km_manhattan",
                                   header = TRUE, sep = ",")

setwd(dir = "~/research/gender_prediction/plots/results/")
generate_label_ratio_plot(svm_l = T,
                         svm_k = T,
                         rf = T,
                         logistic = T,
                         knn = T,
                         cex.axis = 2,
                         cex.lab = 2.2,
                         cex.main = 2.2,
                         cex = 2)
