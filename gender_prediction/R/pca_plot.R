
generate_pca_plot = function(cex.axis,
                             cex.lab,
                             cex.main,
                             cex) {
  filename="pca_explained_variance.png"
  png(filename, width=1900, height=1000, units="px", pointsize = 25)
  par(mar = c(4, 5, 1, 1))
  plot(sa_pca_data$Component_Number, sa_pca_data$Total_Explained_Variance,
       #main="Explained Variance per Principal Components",
       xlab="Number of Components",
       ylab="Total Explain Variance (%)",
       xlim = c(0, 95),
       ylim=c(20,100),
       lab=c(10,10,10),
       cex.main=cex.main,
       cex.lab = cex.lab,
       cex.axis = cex.axis,
       type = "p",
       cex = cex,
       pch = 19,  
       col ="indianred",
       bty="o")
  
  points(eu_pca_data$Component_Number, eu_pca_data$Total_Explained_Variance,
         type = "p",
         cex = cex,
         pch = 19,  
         col ="dodgerblue")
  
  legend(x = "bottomright",
         legend = c("South Asian", "European"),
         col = c("indianred", "dodgerblue"),
         pch = c(19,19),
         cex = cex.main,
         pt.cex = cex,
         bty="n")
  dev.off()
}

############################################

setwd(dir = "~/research/gender_prediction/results/pca/south_asian/")
sa_pca_data = read.csv(file = "cleaned_south_asian_data_a2_c2.csv.explained_var", header = TRUE, sep = ",")
setwd(dir = "~/research/gender_prediction/results/pca/european/")
eu_pca_data = read.csv(file = "cleaned_european_data_a2_c2.csv.explained_var", header = TRUE, sep = ",")
setwd(dir = "~/research/gender_prediction/plots/results/")
generate_pca_plot(cex.axis = 2,
                  cex.lab = 2.2,
                  cex.main = 2.2,
                  cex = 2)
