setwd(dir = "~/research/gender_prediction/results/all/")
full_accuracy_data =read.csv(file = "full-accuracy.csv", header = TRUE, sep = ",")
precision_per_gender_data =read.csv(file = "precision-per-gender.csv", header = TRUE, sep = ",")
confidence_per_gender_data =read.csv(file = "confidence-per-gender.csv", header = TRUE, sep = ",")

# varying train size data
sa_train_size_linear_svm_data = read.csv(file = "varying_train/south_asian_varying_train_linear_svm.csv",
                                         header = TRUE, sep = ",")
sa_train_size_kernel_svm_data = read.csv(file = "varying_train/south_asian_varying_train_kernel_svm.csv",
                                         header = TRUE, sep = ",")
sa_train_size_rf_data = read.csv(file = "varying_train/south_asian_varying_train_rf.csv",
                                 header = TRUE, sep = ",")
eu_train_size_linear_svm_data = read.csv(file = "varying_train/european_varying_train_linear_svm.csv",
                                         header = TRUE, sep = ",")
eu_train_size_kernel_svm_data = read.csv(file = "varying_train/european_varying_train_kernel_svm.csv",
                                         header = TRUE, sep = ",")
eu_train_size_rf_data = read.csv(file = "varying_train/european_varying_train_rf.csv",
                                 header = TRUE, sep = ",")
# feature_selection data
sa_features_linear_svm_data = read.csv(file = "feature_selection/south_asian_linear_svm.csv",
                                       header = TRUE, sep = ",")
sa_features_kernel_svm_data = read.csv(file = "feature_selection/south_asian_kernel_svm.csv",
                                       header = TRUE, sep = ",")
sa_features_rf_data = read.csv(file = "feature_selection/south_asian_rf.csv",
                               header = TRUE, sep = ",")

eu_features_linear_svm_data = read.csv(file = "feature_selection/european_linear_svm.csv",
                                       header = TRUE, sep = ",")
eu_features_kernel_svm_data = read.csv(file = "feature_selection/european_kernel_svm.csv",
                                       header = TRUE, sep = ",")
eu_features_rf_data = read.csv(file = "feature_selection/european_rf.csv",
                               header = TRUE, sep = ",")

# pca data 
sa_pca_data = read.csv(file = "pca/south_asian_explained_var_pca.csv", header = TRUE, sep = ",")
eu_pca_data = read.csv(file = "pca/european_explained_var_pca.csv", header = TRUE, sep = ",")


generate_full_accuracy_plot = function(svm_l, svm_k, rf) {
  legends = character(0)
  legend_colors = character(0)
  filename="full-accuracy.png"
  png(filename, width=1900, height=1400, units="px", pointsize = 25)
  plot(1,
       type="n",
       main='Accuracy with Different Data Subsets',
       xlab='Top Percentage Predicted',
       ylab='Accuracy',
       xlim = c(0, 100),
       ylim= c(0.7, 0.95),
       cex.main=1.2,
       cex.lab = 1.2,
       cex.axis = 1.2)

  if (svm_l) {
    lines(full_accuracy_data$Percentage.predicted,
          full_accuracy_data$EU.SVM.L,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="blue")
    lines(full_accuracy_data$Percentage.predicted,
          full_accuracy_data$SA.SVM.L,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="red")
    legends = append(legends, c("EU-SVM-L", "SA-SVM-L"))
    legend_colors = append(legend_colors, c("blue", "red"))
  }
  
  if (svm_k) {
    lines(full_accuracy_data$Percentage.predicted,
            full_accuracy_data$EU.SVM.K,
            type = "p",
            lwd = 2.5,
            cex = 0.7,
            pch = 19,  
            col="deepskyblue")
    lines(full_accuracy_data$Percentage.predicted,
          full_accuracy_data$SA.SVM.K,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="red4")
    legends = append(legends, c("EU-SVM-K", "SA-SVM-K"))
    legend_colors = append(legend_colors, c("deepskyblue", "red4"))
  }
  
  if (rf) {
    lines(full_accuracy_data$Percentage.predicted,
          full_accuracy_data$EU.RF,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="dodgerblue4")
    lines(full_accuracy_data$Percentage.predicted,
          full_accuracy_data$SA.RF,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="indianred1")
    legends = append(legends, c("EU-RF", "SA-RF"))
    legend_colors = append(legend_colors, c("dodgerblue4", "indianred1"))
  }

  legend(x = "topright",
         legend=legends,
         col=legend_colors,
         pch=c(19,19,19,19,19,19),
         cex=1,
         bty="n")
  dev.off()
}





generate_precision_per_gender_plot = function(svm_l, svm_k, rf) {
  legends = character(0)
  legend_colors = character(0)
  legend_pchs = numeric(0)
  legend_cexs = numeric(0)
  filename="precision-per-gender.png"
  png(filename, width=1900, height=1400, units="px", pointsize = 25)
  plot(1,
       type="n",
       main='Precision Per Gender',
       xlab='Top Percentage Predicted',
       ylab='Precision',
       xlim = c(0, 100),
       ylim= c(70, 95),
       cex.main=1.2)
  
  tmp = precision_per_gender_data$EU.Male.SVM.L.percent.predicted
  EU.Male.SVM.L.percent.predicted = precision_per_gender_data$EU.Male.SVM.L.percent.predicted[tmp > 5]
  EU.Male.SVM.L.Precision = precision_per_gender_data$EU.Male.SVM.L.Precision[tmp > 5]
  
  tmp = precision_per_gender_data$EU.Female.SVM.L.percent.predicted
  EU.Female.SVM.L.percent.predicted = precision_per_gender_data$EU.Female.SVM.L.percent.predicted[tmp > 5]
  EU.Female.SVM.L.Precision = precision_per_gender_data$EU.Female.SVM.L.Precision[tmp > 5]
  
  
  tmp = precision_per_gender_data$EU.Male.SVM.K.percent.predicted
  EU.Male.SVM.K.percent.predicted = precision_per_gender_data$EU.Male.SVM.K.percent.predicted[tmp > 5]
  EU.Male.SVM.K.Precision = precision_per_gender_data$EU.Male.SVM.K.Precision[tmp > 5]
  
  tmp = precision_per_gender_data$EU.Female.SVM.K.percent.predicted
  EU.Female.SVM.K.percent.predicted = precision_per_gender_data$EU.Female.SVM.K.percent.predicted[tmp > 5]
  EU.Female.SVM.K.Precision = precision_per_gender_data$EU.Female.SVM.K.Precision[tmp > 5]
  
  tmp = precision_per_gender_data$EU.Male.RF.percent.predicted
  EU.Male.RF.percent.predicted = precision_per_gender_data$EU.Male.RF.percent.predicted[tmp > 5]
  EU.Male.RF.Precision = precision_per_gender_data$EU.Male.RF.Precision[tmp > 5]
  
  tmp = precision_per_gender_data$EU.Female.RF.percent.predicted
  EU.Female.RF.percent.predicted = precision_per_gender_data$EU.Female.RF.percent.predicted[tmp > 5]
  EU.Female.RF.Precision = precision_per_gender_data$EU.Female.RF.Precision[tmp > 5]
  
  
  tmp = precision_per_gender_data$SA.Male.SVM.L.percent.predicted
  SA.Male.SVM.L.percent.predicted = precision_per_gender_data$SA.Male.SVM.L.percent.predicted[tmp > 25]
  SA.Male.SVM.L.Precision = precision_per_gender_data$SA.Male.SVM.L.Precision[tmp > 25]
  
  tmp = precision_per_gender_data$SA.Female.SVM.L.percent.predicted
  SA.Female.SVM.L.percent.predicted = precision_per_gender_data$SA.Female.SVM.L.percent.predicted[tmp > 25]
  SA.Female.SVM.L.Precision = precision_per_gender_data$SA.Female.SVM.L.Precision[tmp > 25]
  
  
  tmp = precision_per_gender_data$SA.Male.SVM.K.percent.predicted
  SA.Male.SVM.K.percent.predicted = precision_per_gender_data$SA.Male.SVM.K.percent.predicted[tmp > 25]
  SA.Male.SVM.K.Precision = precision_per_gender_data$SA.Male.SVM.K.Precision[tmp > 25]
  
  tmp = precision_per_gender_data$SA.Female.SVM.K.percent.predicted
  SA.Female.SVM.K.percent.predicted = precision_per_gender_data$SA.Female.SVM.K.percent.predicted[tmp > 25]
  SA.Female.SVM.K.Precision = precision_per_gender_data$SA.Female.SVM.K.Precision[tmp > 25]
  
  tmp = precision_per_gender_data$SA.Male.RF.percent.predicted
  SA.Male.RF.percent.predicted = precision_per_gender_data$SA.Male.RF.percent.predicted[tmp > 25]
  SA.Male.RF.Precision = precision_per_gender_data$SA.Male.RF.Precision[tmp > 25]
  
  tmp = precision_per_gender_data$SA.Female.RF.percent.predicted
  SA.Female.RF.percent.predicted = precision_per_gender_data$SA.Female.RF.percent.predicted[tmp > 25]
  SA.Female.RF.Precision = precision_per_gender_data$SA.Female.RF.Precision[tmp > 25]
  
  if (svm_l) {
    lines(EU.Male.SVM.L.percent.predicted,
          EU.Male.SVM.L.Precision,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,
          col="blue")
    lines(EU.Female.SVM.L.percent.predicted,
          EU.Female.SVM.L.Precision,
          type = "p",
          lwd = 2.5,
          cex = 1,
          pch = 17,
          col="blue")
    lines(SA.Male.SVM.L.percent.predicted,
          SA.Male.SVM.L.Precision,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,
          col="red")
    lines(SA.Female.SVM.L.percent.predicted,
          SA.Female.SVM.L.Precision,
          type = "p",
          lwd = 2.5,
          cex = 1,
          pch = 17,
          col="red")
    legends = append(legends, c("EU-Male-SVM-L", "EU-Female-SVM-L", "SA-Male-SVM-L", "SA-Female-SVM-L"))
    legend_colors = append(legend_colors, c("blue", "blue", "red", "red"))
    legend_pchs = append(legend_pchs, c(19,17,19,17))
    legend_cexs = append(legend_cexs, c(1,1.2,1,1.2))
  }
  if (svm_k) {
    lines(EU.Male.SVM.K.percent.predicted,
          EU.Male.SVM.L.Precision, type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,
          col="deepskyblue")
    lines(EU.Female.SVM.K.percent.predicted,
          EU.Female.SVM.K.Precision,
          type = "p",
          lwd = 2.5,
          cex = 1,
          pch = 17,
          col="deepskyblue")
    lines(SA.Male.SVM.K.percent.predicted,
          SA.Male.SVM.K.Precision,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,
          col="red4")
    lines(SA.Female.SVM.K.percent.predicted,
          SA.Female.SVM.K.Precision,
          type = "p",
          lwd = 2.5,
          cex = 1,
          pch = 17,
          col="red4")
    legends = append(legends, c("EU-Male-SVM-K", "EU-Female-SVM-K", "SA-Male-SVM-K", "SA-Female-SVM-K"))
    legend_colors = append(legend_colors, c("deepskyblue", "deepskyblue", "red4", "red4"))
    legend_pchs = append(legend_pchs, c(19,17,19,17))
    legend_cexs = append(legend_cexs, c(1,1.2,1,1.2))
  }
  if (rf) {
    lines(EU.Male.RF.percent.predicted,
          EU.Male.RF.Precision,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,
          col="dodgerblue4")
    lines(EU.Female.RF.percent.predicted,
          EU.Female.RF.Precision,
          type = "p",
          lwd = 2.5,
          cex = 1,
          pch = 17,
          col="dodgerblue4")
    lines(SA.Male.RF.percent.predicted,
          SA.Male.RF.Precision,
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,
          col="indianred1")
    lines(SA.Female.RF.percent.predicted,
          SA.Female.RF.Precision,
          type = "p",
          lwd = 2.5,
          cex = 1,
          pch = 17,
          col="indianred1")
    legends = append(legends, c("EU-Male-RF", "EU-Female-RF", "SA-Male-RF", "SA-Female-RF"))
    legend_colors = append(legend_colors, c("dodgerblue4", "dodgerblue4", "indianred1", "indianred1"))
    legend_pchs = append(legend_pchs, c(19,17,19,17))
    legend_cexs = append(legend_cexs, c(1,1.2,1,1.2))
  }
  legend(x = "topright",
         legend=legends,
         col=legend_colors,
         pch=legend_pchs,
         cex=1,
         pt.cex=legend_cexs,
         bty="n")
  dev.off()
}





generate_confidence_per_gender_plot = function() {
  filename="confidence-per-gender.png"
  png(filename, width=1900, height=1400, units="px", pointsize = 25)
  plot(confidence_per_gender_data$EU.SVM.L.Min.Conf,
       confidence_per_gender_data$EU.Male.SVM.L.percent.predicted,
       main='Confidence Per Gender',
       xlab='Confidence',
       ylab='Percentage of Predictions',
       col="blue",
       type = "p",
       lwd = 2.5,
       cex = 0.7,
       pch = 19,  
       xlim = c(50, 100),
       ylim= c(0, 100),
       cex.main=1.2)
  lines(confidence_per_gender_data$EU.SVM.K.Min.Conf,
        confidence_per_gender_data$EU.Male.SVM.L.percent.predicted, type = "p",
        lwd = 2.5,
        cex = 0.7,
        pch = 19,  
        col="deepskyblue")
  lines(confidence_per_gender_data$EU.RF.Min.Conf,
        confidence_per_gender_data$EU.Male.RF.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 0.7,
        pch = 19,  
        col="dodgerblue4")
  
  lines(confidence_per_gender_data$SA.SVM.L.Min.Conf,
        confidence_per_gender_data$SA.Male.SVM.L.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 0.7,
        pch = 19,  
        col="red")
  lines(confidence_per_gender_data$SA.SVM.K.Min.Conf,
        confidence_per_gender_data$SA.Male.SVM.K.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 0.7,
        pch = 19,  
        col="red4")
  lines(confidence_per_gender_data$SA.RF.Min.Conf,
        confidence_per_gender_data$SA.Male.RF.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 0.7,
        pch = 19,  
        col="indianred1")
  
  lines(confidence_per_gender_data$EU.SVM.L.Min.Conf,
        confidence_per_gender_data$EU.Female.SVM.L.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 1,
        pch = 17,  
        col="blue")
  lines(confidence_per_gender_data$EU.SVM.K.Min.Conf,
        confidence_per_gender_data$EU.Female.SVM.K.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 1,
        pch = 17,  
        col="deepskyblue")
  lines(confidence_per_gender_data$EU.RF.Min.Conf,
        confidence_per_gender_data$EU.Female.RF.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 1,
        pch = 17,  
        col="dodgerblue4")
  
  lines(confidence_per_gender_data$SA.SVM.L.Min.Conf,
        confidence_per_gender_data$SA.Female.SVM.L.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 1,
        pch = 17,  
        col="red")
  lines(confidence_per_gender_data$SA.SVM.K.Min.Conf,
        confidence_per_gender_data$SA.Female.SVM.K.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 1,
        pch = 17,  
        col="red4")
  lines(confidence_per_gender_data$SA.RF.Min.Conf,
        confidence_per_gender_data$SA.Female.RF.percent.predicted,
        type = "p",
        lwd = 2.5,
        cex = 1,
        pch = 17,  
        col="indianred1")
  
  legend(x = "topright",
         legend=c("EU-Male-SVM-L", "EU-Male-SVM-K", "EU-Male-RF",
                  "SA-Male-SVM-L", "SA-Male-SVM-K", "SA-Male-RF",
                  "EU-Female-SVM-L", "EU-Female-SVM-K", "EU-Female-RF",
                  "SA-Female-SVM-L", "SA-Female-SVM-K", "SA-Female-RF"),
         col=c("blue", "deepskyblue", "dodgerblue4",
               "red", "red4", "indianred1",
               "blue", "deepskyblue", "dodgerblue4",
               "red", "red4", "indianred1"),
         pch=c(19,19,19,
               19,19,19,
               17,17,17,
               17,17,17),
         cex=1,
         pt.cex=c(1,1,1,
                  1,1,1,
                  1.2,1.2,1.2,
                  1.2,1.2,1.2),
         bty="n")
  dev.off()
}


capitalize = function(x) {
  s = strsplit(x, " ")[[1]]
  s1 = paste(toupper(substring(s, 1, 1)), substring(s, 2), sep = "", collapse = " ")
  return(s1)
}


generate_feature_selection_plot = function(svm_l, svm_k, rf, column) {
  legends = character(0)
  legend_colors = character(0)
  filename=paste("feature-selection-", column, ".png", sep="")
  title = paste(capitalize(column), "vs Top Feature Selection", sep=" ")
  min_y=200
  max_y=0
  if (svm_l) {
    min_y = min(sa_features_linear_svm_data[[column]],
                eu_features_linear_svm_data[[column]],
                min_y)
    max_y = max(sa_features_linear_svm_data[[column]],
                eu_features_linear_svm_data[[column]],
                max_y)
  }
  if (svm_k) {
    min_y = min(sa_features_kernel_svm_data[[column]],
                eu_features_kernel_svm_data[[column]],
                min_y)
    max_y = max(sa_features_kernel_svm_data[[column]],
                eu_features_kernel_svm_data[[column]],
                max_y)
  }
  if (rf) {
    min_y = min(sa_features_rf_data[[column]],
                eu_features_rf_data[[column]],
                min_y)
    max_y = max(sa_features_rf_data[[column]],
                eu_features_rf_data[[column]],
                max_y)
  }
  
  png(filename, width=1900, height=1400, units="px", pointsize = 25)
  plot(1,
       type="n",
       main=title,
       xlab='Top Features Selected',
       ylab=capitalize(column),
       xlim = c(0, 100),
       ylim= c(min_y*0.95, max_y*1.05),
       cex.main=1.2)
  
  if (svm_l) {
    lines(eu_features_linear_svm_data$num_features_selected,
          eu_features_linear_svm_data[[column]],
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="blue")
    lines(sa_features_linear_svm_data$num_features_selected,
          sa_features_linear_svm_data[[column]],
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="red")
    legends = append(legends, c("EU-SVM-L", "SA-SVM-L"))
    legend_colors = append(legend_colors, c("blue", "red"))
  }
  if (svm_k) {
    lines(eu_features_kernel_svm_data$num_features_selected,
          eu_features_kernel_svm_data[[column]],
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="deepskyblue")
    lines(sa_features_kernel_svm_data$num_features_selected,
          sa_features_kernel_svm_data[[column]],
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="red4")
    legends = append(legends, c("EU-SVM-K", "SA-SVM-K"))
    legend_colors = append(legend_colors, c("deepskyblue", "red4"))
  }
  if (rf) {
    lines(eu_features_rf_data$num_features_selected,
          eu_features_rf_data[[column]],
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="dodgerblue4")
    lines(sa_features_rf_data$num_features_selected,
          sa_features_rf_data[[column]],
          type = "p",
          lwd = 2.5,
          cex = 0.7,
          pch = 19,  
          col="indianred1")
    legends = append(legends, c("EU-RF", "SA-RF"))
    legend_colors = append(legend_colors, c("dodgerblue4", "indianred1"))
  }
  legend(x = "bottomright",
         legend=legends,
         col=legend_colors,
         pch=c(19,19,19,19,19,19),
         cex=1,
         bty="n")
  dev.off()
}



generate_train_size_plot = function(svm_l, svm_k, rf) {
  legends = character(0)
  legend_colors = character(0)
  filename="south_asian_varying_train_size.png"
  png(filename, width=1900, height=1400, units="px", pointsize = 25)
  title = paste("Effect of Training size on South Asian Data\n",
                "Test size: ", mean(sa_train_size_linear_svm_data$test_size))
  plot(1,
       type="n",
       main = title,
       xlab="Training Size",
       ylab="Test Accuracy",
       xlim = c(500, 20000),
       ylim=c(67,75),
       lab=c(30,10,10),
       cex.main=1.5,
       cex.lab = 1.3,
       cex.axis = 1.3)
  
  if (svm_l) {
    lines(sa_train_size_linear_svm_data$train_size,
          sa_train_size_linear_svm_data$test_accuracy,
          type = "p",
          cex = 1.1,
          pch = 19,  
          col ="indianred")
    legends = append(legends, c("Linear SVM"))
    legend_colors = append(legend_colors, c("indianred"))
  }
  if (svm_k) {
    lines(sa_train_size_kernel_svm_data$train_size,
          sa_train_size_kernel_svm_data$test_accuracy,
          type = "p",
          cex = 1.1,
          pch = 19,  
          col="forestgreen")
    legends = append(legends, c("Kernel SVM"))
    legend_colors = append(legend_colors, c("forestgreen"))
  }
  if (rf) {
    lines(sa_train_size_rf_data$train_size,
          sa_train_size_rf_data$test_accuracy,
          type = "p",
          cex = 1.1,
          pch = 19,  
          col="dodgerblue")
    legends = append(legends, c("RF"))
    legend_colors = append(legend_colors, c("dodgerblue"))
  }

  legend(x = "bottomright",
         legend=legends,
         col=legend_colors,
         pch=c(19,19,19,19,19,19),
         cex=1.4,
         pt.cex=1.1,
         bty="n")
  dev.off()
  
  # plot european data now  
  legends = character(0)
  legend_colors = character(0)
  filename="european_varying_train_size.png"
  png(filename, width=1900, height=1400, units="px", pointsize = 25)
  title = paste("Effect of Training size on European Data\n",
                "Test size: ", mean(eu_train_size_linear_svm_data$test_size))
  plot(1,
       type="n",
       main = title,
       xlab="Training Size",
       ylab="Test Accuracy",
       xlim = c(500, 30000),
       ylim=c(67,74),
       lab=c(30,10,10),
       cex.main=1.5,
       cex.lab = 1.3,
       cex.axis = 1.3)
  
  if (svm_l) {
    lines(eu_train_size_linear_svm_data$train_size,
          eu_train_size_linear_svm_data$test_accuracy,
          type = "p",
          cex = 1.1,
          pch = 19,  
          col ="indianred")
    legends = append(legends, c("Linear SVM"))
    legend_colors = append(legend_colors, c("indianred"))
  }
  if (svm_k) {
    lines(eu_train_size_kernel_svm_data$train_size,
          eu_train_size_kernel_svm_data$test_accuracy,
          type = "p",
          cex = 1.1,
          pch = 19,  
          col="forestgreen")
    legends = append(legends, c("Kernel SVM"))
    legend_colors = append(legend_colors, c("forestgreen"))
  }
  if (rf) {
    lines(eu_train_size_rf_data$train_size,
          eu_train_size_rf_data$test_accuracy,
          type = "p",
          cex = 1.1,
          pch = 19,  
          col="dodgerblue")
    legends = append(legends, c("RF"))
    legend_colors = append(legend_colors, c("dodgerblue"))
  }

  legend(x = "bottomright",
         legend=legends,
         col=legend_colors,
         pch=c(19,19,19,19,19,19),
         cex=1.4,
         pt.cex=1.1,
         bty="n")
  dev.off()
}
  
generate_pca_plot = function() {
  filename="pca_explained_variance.png"
  png(filename, width=1900, height=1400, units="px", pointsize = 25)
  par(mar = c(4, 4.5, 1, 1))
  plot(sa_pca_data$Component_Number, sa_pca_data$Total_Explained_Variance,
       #main="Explained Variance per Principal Components",
       xlab="Number of Components",
       ylab="Total Explain Variance (%)",
       xlim = c(0, 100),
       ylim=c(20,100),
       lab=c(10,10,10),
       cex.main=1.9,
       cex.lab = 1.7,
       cex.axis = 1.7,
       type = "p",
       cex = 1.7,
       pch = 19,  
       col ="indianred")
  
  points(eu_pca_data$Component_Number, eu_pca_data$Total_Explained_Variance,
         type = "p",
         cex = 1.7,
         pch = 19,  
         col ="dodgerblue")
  
  legend(x = "bottomright",
         legend=c("South Asian", "European"),
         col=c("indianred", "dodgerblue"),
         pch=c(19,19),
         cex=1.9,
         pt.cex=1.7,
         bty="n")
  dev.off()
}


# set the dir for output plots
setwd(dir = "~/research/gender_prediction/plots/results/")

# PLOT FULL ACCURACY
generate_full_accuracy_plot(svm_l=F, svm_k=T, rf=T)
# PLOT PER GENDER FULL ACCURACY
generate_precision_per_gender_plot(svm_l=F, svm_k=T, rf=F)
# PLOT PER GENDER CONFIDENCE 
generate_confidence_per_gender_plot()
# PLOT FEATURE SELECTION
generate_feature_selection_plot(svm_l=T, svm_k=T, rf=F, column="accuracy")
# PLOT VARYING TRAIN SIZE
generate_train_size_plot(svm_l=T, svm_k=T, rf=T)
# PLOT EXPLAINED VAR BY PCA
generate_pca_plot()
