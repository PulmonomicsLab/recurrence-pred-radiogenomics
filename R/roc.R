#### Radiological - SMOTE - Inception ####

file_name <- "output/roc/radiological/roc_summary_smote_inception.tsv"
data <- read.csv(file_name, sep = "\t")
# print(data)
print(c(ncol(data), nrow(data)))
cols_reg_cv <- c("SVM (AUC=0.965)", "RF (AUC=0.949)", "MLP (AUC=0.960)")
cols_reg_blind <- c("SVM (AUC=0.933)", "RF (AUC=0.927)", "MLP (AUC=0.935)")
x <- as.array(data$FPR)
svr_cv_y <- as.array(data[, 2])
rf_cv_y <- as.array(data[, 3])
mlpr_cv_y <- as.array(data[, 4])
svr_blind_y <- as.array(data[, 5])
rf_blind_y <- as.array(data[, 6])
mlpr_blind_y <- as.array(data[, 7])
lwd <- c(1.5, 2, 2)
lty <- c(1, 3, 2)
# par(mfrow = c(1, 2))

# # svg("roc_plot_radiological_smote_inception_cv.svg", width=8, height=8)
# plot(x, svr_cv_y, xlab = "False positive rate", ylab = "True positive rate",
#      type = "l", lwd = lwd[1], lty = lty[1], cex.lab=1.2)
# lines(x, rf_cv_y, type = "l", lwd = lwd[2], lty = lty[2])
# lines(x, mlpr_cv_y, type = "l", lwd = lwd[3], lty = lty[3])
# legend(x = "bottomright", lty = lty, lwd=lwd, cex=1.1, legend=cols_reg_cv)
# # title("a.", adj=0, line=0.5)
# # dev.off()
# # svg("roc_plot_radiological_smote_inception_blind.svg", width=8, height=8)
# plot(x, svr_cv_y, xlab = "False positive rate", ylab = "True positive rate", 
#      type = "l", lwd = lwd[1], lty = lty[1], cex.lab=1.2)
# lines(x, rf_cv_y, type = "l", lwd = lwd[2], lty = lty[2])
# lines(x, mlpr_cv_y, type = "l", lwd = lwd[3], lty = lty[3])
# legend(x = "bottomright", lty = lty, lwd=lwd, cex=1.1, legend=cols_reg_blind)
# # title("b.", adj=0, line=0.5)
# # dev.off()

#### Combined - SMOTE - Xception ####

file_name <- "output/roc/combined/roc_summary_smote_xception.tsv"
data <- read.csv(file_name, sep = "\t")
# print(data)
print(c(ncol(data), nrow(data)))
cols_reg_cv <- c("SVM (AUC=0.991)", "RF (AUC=0.995)", "MLP (AUC=0.995)")
cols_reg_blind <- c("SVM (AUC=0.978)", "RF (AUC=1.000)", "MLP (AUC=0.989)")
x <- as.array(data$FPR)
svr_cv_y <- as.array(data[, 2])
rf_cv_y <- as.array(data[, 3])
mlpr_cv_y <- as.array(data[, 4])
svr_blind_y <- as.array(data[, 5])
rf_blind_y <- as.array(data[, 6])
mlpr_blind_y <- as.array(data[, 7])
lwd <- c(1.5, 2, 2)
lty <- c(1, 3, 2)
# par(mfrow = c(1, 2))

# # svg("roc_plot_combined_smote_xception_cv.svg", width=8, height=8)
# plot(x, svr_cv_y, xlab = "False positive rate", ylab = "True positive rate",
#      type = "l", lwd = lwd[1], lty = lty[1], cex.lab=1.2)
# lines(x, rf_cv_y, type = "l", lwd = lwd[2], lty = lty[2])
# lines(x, mlpr_cv_y, type = "l", lwd = lwd[3], lty = lty[3])
# legend(x = "bottomright", lty = lty, lwd=lwd, cex=1.1, legend=cols_reg_cv)
# # title("a.", adj=0, line=0.5)
# # dev.off()
# # svg("roc_plot_combined_smote_xception_blind.svg", width=8, height=8)
# plot(x, svr_cv_y, xlab = "False positive rate", ylab = "True positive rate", 
#      type = "l", lwd = lwd[1], lty = lty[1], cex.lab=1.2)
# lines(x, rf_cv_y, type = "l", lwd = lwd[2], lty = lty[2])
# lines(x, mlpr_cv_y, type = "l", lwd = lwd[3], lty = lty[3])
# legend(x = "bottomright", lty = lty, lwd=lwd, cex=1.1, legend=cols_reg_blind)
# # title("b.", adj=0, line=0.5)
# # dev.off()

#### Summary ####

file_name <- "output/roc/roc_summary.tsv"
data <- read.csv(file_name, sep = "\t")
# print(data)
print(c(ncol(data), nrow(data)))
cols_reg_blind <- c("Radiological (AUC=0.93)", "Genomic (AUC=0.93)", "Combined (AUC=0.98)")
x <- as.array(data$FPR)
radiological_blind_y <- as.array(data[, 2])
genomic_blind_y <- as.array(data[, 3])
combined_blind_y <- as.array(data[, 4])
lwd <- c(2, 2, 1.5)
lty <- c(2, 3, 1)
# par(mfrow = c(1, 2))

# # svg("roc_plot_summary_blind.svg", width=8, height=8)
# plot(x, radiological_blind_y, xlab = "False positive rate", ylab = "True positive rate",
#      type = "l", lwd = lwd[1], lty = lty[1], cex.lab=1.4)
# lines(x, genomic_blind_y, type = "l", lwd = lwd[2], lty = lty[2])
# lines(x, combined_blind_y, type = "l", lwd = lwd[3], lty = lty[3])
# legend(x = "bottomright", lty = lty, lwd=lwd, cex=1.1, legend=cols_reg_blind)
# # dev.off()

#### Summary (Radiological + Combined) ####

file_name <- "output/roc/roc_summary.tsv"
data <- read.csv(file_name, sep = "\t")
# print(data)
print(c(ncol(data), nrow(data)))
cols_reg_blind <- c("Optimal radiological model (AUC=0.933)", "Optimal radiogenomic model (AUC=0.978)")
x <- as.array(data$FPR)
radiological_blind_y <- as.array(data[, 2])
combined_blind_y <- as.array(data[, 4])
lwd <- c(2.5, 1.5)
lty <- c(2, 1)
# par(mfrow = c(1, 2))

# # svg("roc_plot_summary_blind_2.svg", width=8, height=8)
# plot(x, radiological_blind_y, xlab = "False positive rate", ylab = "True positive rate",
#      type = "l", lwd = lwd[1], lty = lty[1], cex.lab=1.4)
# lines(x, combined_blind_y, type = "l", lwd = lwd[2], lty = lty[2])
# lines(c(0, 1), c(0, 1), type = "l", lwd = 1, lty = 3)
# legend(x = "bottomright", lty = lty, lwd=lwd, cex=1.1, legend=cols_reg_blind)
# # dev.off()
