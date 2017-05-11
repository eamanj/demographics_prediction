################################################
setwd(dir = "~/research/gender_prediction/data")
south_asian_data = read.csv(file = "cleaned_south_asian_data.csv", header = TRUE, sep = ",")
setwd(dir = "~/research/gender_prediction/south_asian_plots/")
num_data_points = nrow(south_asian_data)

numerics = sapply(south_asian_data, is.numeric)
for (column in names(south_asian_data)) {
  column_data = south_asian_data[[column]]
  if (!numerics[column]) {
    datatable = table(column_data)
    extra_info = paste("num_data_points =", length(column_data), "\n",
                       "max =", signif(max(datatable), 4),
                       "min =", signif(min(datatable), 4), 
                       "median =", signif(median(datatable), 4),
                       "mean =", signif(mean(datatable), 4))
    title = paste("distribution of", column, "indicator")
    filename=paste0(column, ".png")
    png(filename, width=700, height=700, units="px", pointsize = 12)
    par(mai=c(1.5,1.5,2,1.5))
    barplot(datatable,
            main = title,
            xlab = column,
            ylab = "Frequency")
    mtext(extra_info, side = 3, line =0.5)
    dev.off()
  } else {
    num_unique_vals = length(unique(column_data))
    extra_info = paste("num_data_points =", length(column_data), "\n",
                       "max =", signif(max(column_data), 4),
                       "min =", signif(min(column_data), 4), 
                       "median =", signif(median(column_data), 4),
                       "mean =", signif(mean(column_data), 4))
    num_bins = max(floor(num_unique_vals*0.3), 30)
    histogram = hist(column_data, breaks=num_bins, plot=F)
    
    title = paste("distribution of", column, "indicator")
    filename=paste0(column, ".png")
    png(filename, width=700, height=700, units="px", pointsize = 12)
    par(mai=c(1.5,1.5,2,1.5))
    num_bins = max(floor(num_unique_vals*0.3), 30)
    hist(column_data, main=title, xlab=column, ylab="Frequency", breaks=num_bins)
    mtext(extra_info, side = 3, line =0.5)
    dev.off()
    
    # now plot scaled version1
    title = paste("distribution of", column, "indicator (y scaled 20%)")
    filename=paste0(column, "_y_20_scaled.png")
    png(filename, width=700, height=700, units="px", pointsize = 12)
    par(mai=c(1.5,1.5,2,1.5))
    # dont let a single bin contain more than 20%
    max_y = min(0.2*num_data_points, max(histogram$counts))
    hist(column_data, main=title, xlab=column, ylab="Frequency", breaks=num_bins, ylim=c(0,max_y))
    mtext(extra_info, side = 3, line =0.5)
    dev.off()
    
    # now plot scaled version2
    title = paste("distribution of", column, "indicator (y log scale)")
    filename=paste0(column, "_y_log_scaled.png")
    png(filename, width=700, height=700, units="px", pointsize = 12)
    par(mai=c(1.5,1.5,2,1.5))
    plot(histogram$mids, histogram$count, log="y", type='h',
         lwd=2, lend=2,
         main=title, xlab=column, ylab="Frequency")
    mtext(extra_info, side = 3, line =0.5)
    dev.off()
  }
}
column="duration_of_calls__call__kurtosis__mean"

