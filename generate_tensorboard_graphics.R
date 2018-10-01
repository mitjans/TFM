fix_dense_low_cnn <- read.csv2("run_4cnn_128-8_256-6_512-4_1024-2_3pooling_3_3_2_4hdd_2048_1024_512_32_fix_dense_low_cnn-tag-loss.csv", sep = ",", stringsAsFactors = F)
fix_dense_low_cnn$Value <- as.numeric(fix_dense_low_cnn$Value)

fix_cnn_low_dense <- read.csv2("run_8cnn_128-8_128-8_256-6_256-6_512-4_512-4_1024-2_1024-2_3pooling_3_3_2_2hdd_512_32_fix_cnn_low_dense-tag-loss.csv", sep = ",", stringsAsFactors = F)
fix_cnn_low_dense$Value <- as.numeric(fix_cnn_low_dense$Value)

fix_cnn_high_dense <- read.csv2("run_8cnn_128-8_128-8_256-6_256-6_512-4_512-4_1024-2_1024-2_3pooling_3_3_2_8hdd_8192_4096_2048_1024_512_256_128_32_fix_cnn_high_dense-tag-loss.csv", sep = ",", stringsAsFactors = F)
fix_cnn_high_dense$Value <- as.numeric(fix_cnn_high_dense$Value)

fix_cnn_fix_dense <- read.csv2("run_8cnn_128-8_128-8_256-6_256-6_512-4_512-4_1024-2_1024-2_3pooling_3_3_2_4hdd_2048_1024_512_32_fix_cnn_fix_dense-tag-loss.csv", sep = ",", stringsAsFactors = F)
fix_cnn_fix_dense$Value <- as.numeric(fix_cnn_fix_dense$Value)

l <- list("Fix convolutional layers, fix dense layers" = data.frame(x=seq(length(fix_cnn_fix_dense$Value)), y=fix_cnn_fix_dense$Value),
                 "Fix convolutional layers, less dense layers" = data.frame(x=seq(length(fix_cnn_low_dense$Value)), y=fix_cnn_low_dense$Value),
                 "Fix convolutional layers, more dense layers" = data.frame(x=seq(length(fix_cnn_high_dense$Value)),y=fix_cnn_high_dense$Value),
                 "Fix dense layers, less convolutional layers" = data.frame(x=seq(length(fix_dense_low_cnn$Value)), y=fix_dense_low_cnn$Value))

ggplot(bind_rows(l, .id="Model"), aes(x, y, colour=Model)) + 
  geom_line(size = 0.2) + 
  geom_smooth(se=F) + 
  theme_bw() +
  theme(legend.position = "bottom", legend.direction = "vertical", legend.text = element_text(size=15))
  ylab("Accuracy") + 
  xlab("Step")
ggsave("all_cost_functions.eps", device = "eps")

setEPS()
postscript("low_cnn_fix_dense-cost_function.eps")
plot(f$Value, pch = 16, ylab = "Loss", xlab = "Step", main = "Less convolutional layers, fix dense layers")
lines(f$Value)
grid()
dev.off()
