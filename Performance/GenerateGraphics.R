suppressWarnings(suppressMessages(library(rjson)))
suppressWarnings(suppressMessages(library(reshape2)))
suppressWarnings(suppressMessages(library(ggplot2)))
suppressWarnings(suppressMessages(library(grid)))
suppressWarnings(suppressMessages(library(gridExtra)))
suppressWarnings(suppressMessages(library(cowplot)))
suppressWarnings(suppressMessages(library(plyr)))

base_wd = "/Users/carlesm/TFM/Data/"
graphs_wd = "/Users/carlesm/TFM/Graphs/"
setwd(base_wd)

plots <- list()

cat("[R] Starting....\n")

models <- dir()
i <- 0
for (model in dir()) {
  i <- i + 1
  cat(paste0("[R] Working with model '", model, "'\t\t[", i, "/", length(models), "]\n"))
  
  if (dir.exists(paste0(graphs_wd, model))) {
    cat("[R]\t Resuming model...")
  } else {
    cat("[R]\tCreating dirs...\n")
  }
  
  setwd(model)
  
  dir.create(paste0(graphs_wd, model), showWarnings = F)
  dir.create(paste0(graphs_wd, model, "/Global"), showWarnings = F)
  dir.create(paste0(graphs_wd, model, "/Distribution"), showWarnings = F)
  dir.create(paste0(graphs_wd, model, "/Random"), showWarnings = F)
  dir.create(paste0(graphs_wd, model, "/Pawn"), showWarnings = F)
  dir.create(paste0(graphs_wd, model, "/Rook"), showWarnings = F)
  dir.create(paste0(graphs_wd, model, "/Knight"), showWarnings = F)
  dir.create(paste0(graphs_wd, model, "/Bishop"), showWarnings = F)
  dir.create(paste0(graphs_wd, model, "/Queen"), showWarnings = F)
  dir.create(paste0(graphs_wd, model, "/King"), showWarnings = F)
  
  # Indicators
  figure_names <-  c("PAWN", "ROOK", "KNIGHT", "BISHOP", "QUEEN", "KING")
  correct_frequencies = setNames(vector("list", length(figure_names)), figure_names)
  correct_destinations = setNames(vector("list", length(figure_names)), figure_names)
  
  files <- dir()

  for (json_file in dir()) {
    if (json_file == "Rplots.pdf") next
    number <- as.numeric(strsplit(strsplit(json_file, "_")[[1]][[2]], "\\.")[[1]][[1]])
    cat(paste0("\r[R]\tWorking with frame '", json_file, "'\t[", number + 1, "/", length(files), "]"))
    
    # Check if frame already computed
    if (file.exists(paste0(graphs_wd, model, "/Global/global_", sprintf("%010d.png", number)))) next
    
    # Read file
    contents = readLines(json_file)
    data <- fromJSON(contents)
    
    # Evaluate each figure separately
    for (figure in seq(1, length(names(data)))) {
      m <- data.frame(t(data.frame(data[figure])))
      if (nrow(m) == 0) next
      rownames(m) <- seq(0, nrow(m) - 1)
      colnames(m) <- c("Y", "X")
      
      base_df <- expand.grid(seq(-7, 7), seq(-7, 7))
      colnames(base_df) <- c("X", "Y")
      
      z <- melt(table(m))
      z$value <- z$value/sum(z$value)
      z[z$value == 0, "value"] <- NA
      z <- merge(base_df, z, all=T)
      
      if (figure == 2) {
        frames <- data.frame(X = c(0, 0, 1, -1), Y = c(1, 2, 1, 1))
        path <- paste0(graphs_wd, model, "/Pawn")
        title <- "PAWN"
      } else if (figure == 3) {
        frames <- data.frame(X = c(seq(-7, 7), rep(0, 15)), Y = c(rep(0, 15), seq(-7, 7)))
        path <- paste0(graphs_wd, model, "/Rook")
        title <- "ROOK"
      } else if (figure == 4) {
        frames <- data.frame(X = c(-2, -2, -1, -1, 1, 1, 2, 2), Y = c(1, -1, 2, -2, 2, -2, 1, -1))
        path <- paste0(graphs_wd, model, "/Knight")
        title <- "KNIGHT"
      } else if (figure == 5) {
        frames <- data.frame(X = c(seq(-7, 7), seq(7, -7)), Y = c(seq(7, -7), seq(7, -7)))
        path <- paste0(graphs_wd, model, "/Bishop")
        title <- "BISHOP"
      } else if (figure == 6) {
        frames <- data.frame(X = c(c(seq(-7, 7), rep(0, 15)),
                                   c(seq(-7, 7), seq(7, -7))), 
                             Y = c(c(rep(0, 15), seq(-7, 7)),
                                   c(seq(7, -7), seq(7, -7))))
        path <- paste0(graphs_wd, model, "/Queen")
        title <- "QUEEN"
      } else if (figure == 7) {
        frames <- data.frame(X = c(-2, -1, 0, 1, 2, 0, 0, 1, -1, 1, -1), Y = c(0, 0, 0, 0, 0, 1, -1, 1, 1, -1, -1))
        path <- paste0(graphs_wd, model, "/King")
        title <- "KING"
      } else {
        frames <- data.frame(X = 0, Y = 0)
        path <- paste0(graphs_wd, model, "/Random")
        title <- "RANDOM"
      }
      
      # Remove 0, 0
      frames <- frames[!(frames$X == 0 & frames$Y == 0), ]
      
      correct_movements <- merge(frames, z)
      temp <- rbind.fill(frames, z[,-3])
      incorrect_movements <- merge(temp[!(duplicated(temp) | duplicated(temp, fromLast = T)),], z)

      p <- ggplot(data = z) +
        # Correct moves
        geom_tile(data = correct_movements, colour="black", aes(x = X, y = Y, fill = pmin(value + 0.08, 1))) +
        # Incorrect moves
        geom_tile(data = incorrect_movements, colour="black", aes(x = X, y = Y, fill = pmax(value*-1 - 0.08, -1))) +
        
        scale_fill_gradient2(low = "indianred", mid = "white", high = "palegreen", na.value = "gray", 
                             guide = guide_colorbar(title = "Frequency", title.position = "bottom", 
                                                    title.hjust = 0.5, barwidth = 15), limits=c(-1, 1), labels = c(1, 0.5, 0, 0.5, 1)) +
        coord_equal() + 
        theme_bw(base_size = 14) + 
        scale_x_discrete(limits = -7:7, expand = c(0.00265, 0.00265)) +
        scale_y_discrete(limits = -7:7, expand = c(0.00265, 0.00265)) +
        theme(plot.title = element_text(face="bold", hjust = 0.5, size = 20, margin = margin(0, 0, 30, 0))) +
        theme(axis.title.x = element_text(margin = margin(10, 0, 0, 0))) +
        theme(axis.title.y = element_text(margin = margin(0, 0, 0, 0))) +
        theme(axis.title.y = element_text(angle = 0, vjust = 0.5)) +
        theme(plot.margin = unit(c(1,1,1,1), "cm")) +
        theme(legend.background = element_rect(fill = NA, colour = "black", linetype = "solid", size = 0.2)) +
        theme(legend.margin = margin(10, 10, 10, 10)) + 
        theme(legend.position = "bottom") +
        ggtitle(label = paste(title, number)) +
        geom_rect(data = frames, fill = NA, colour = "black", size = 0.6, aes(xmin=X-0.5, xmax=X+0.5, ymin=Y-0.5, ymax=Y+0.5)) +
        geom_rect(data = data.frame("X" = 0, "Y" = 0), fill = NA, colour = "navy", size = 1, aes(xmin=X-0.5, xmax=X+0.5, ymin=Y-0.5, ymax=Y+0.5))
      
      if (figure != 1) {
        plots <- c(plots, list(p))
      }
      
      p <- p + geom_text(data = correct_movements, aes(x = X, y = Y, label=round(value,2)), na.rm = T) +
        geom_text(data = incorrect_movements, aes(x = X, y = Y, label=round(value, 2)), na.rm = T)
      
      if (figure == 1) {
        ggsave(paste0(title, "_", sprintf("%010d", number), '.png'), p,
               width = 12, height = 10, limitsize = F,  path = path, dpi = 150)
        next
      }
        
      
      #### INDICATORS ####
      # Correct Frequency
      frame_correct_frequency <- sum(correct_movements$value, na.rm = T)
      correct_frequencies[[figure - 1]] <- c(correct_frequencies[[figure - 1]], frame_correct_frequency)
      
      # Correct destinations
      frame_correct_destinations <- sum(!is.na(correct_movements$value)) / nrow(unique(m))
      correct_destinations[[figure - 1]] <- c(correct_destinations[[figure - 1]], frame_correct_destinations)
      
      p2 <- qplot(y = correct_frequencies[[figure - 1]], ylab = "Frequency", xlab = "", main = "Correct\nmovements") + 
          geom_bar(stat = "identity", fill = colorRampPalette(c("red3", "palegreen"))(101)[frame_correct_frequency*100 + 1], width = 2) + 
          scale_x_discrete() + 
          theme_bw(base_size = 14) + 
          scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1), expand = c(0, 0)) + 
          theme(axis.title.x = element_text(margin = margin(20, 0, 200, 0))) + 
          theme(axis.title.y = element_blank()) + 
          theme(panel.border = element_blank()) +
          theme(plot.title = element_text(face="bold", hjust = 0.5, size = 15, margin = margin(200, 0, 30, 0)))

      p3 <- qplot(y = correct_destinations[[figure - 1]], ylab = "Frequency", xlab = "", main = "Correct\ndestinations") + 
        geom_bar(stat = "identity", fill = colorRampPalette(c("red3", "palegreen"))(101)[frame_correct_destinations*100 + 1], width = 2) + 
        scale_x_discrete() + 
        theme_bw(base_size = 14) + 
        scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1), expand = c(0, 0)) + 
        theme(axis.title.x = element_text(margin = margin(20, 0, 200, 0))) + 
        theme(axis.title.y = element_blank()) + 
        theme(panel.border = element_blank()) +
        theme(plot.title = element_text(face="bold", hjust = 0.5, size = 15, margin = margin(200, 0, 30, 0)))
      
      ggsave(paste0(title, "_", sprintf("%010d", number), '.png'), plot_grid(p2, p, p3, ncol = 3, rel_widths = c(0.2, 0.6, 0.2), align = "v"),
             width = 12, height = 10, limitsize = F,  path = path, dpi = 150)
    }
    
    if (length(plots) == 6) {
      ggsave(paste0("global_", sprintf("%010d", number), ".png"), do.call(arrangeGrob, c(plots, list(nrow = 2),
                                                                       list(top = textGrob(paste("Global", number),
                                                                                           vjust = 1, gp = gpar(fontface = 3, fontsize = 30))))),
             width = 45.73, height = 32.35, path = paste0(graphs_wd, model, "/Global"), units = "cm",  dpi = 150, limitsize = F)
    }
    
    plots <- list()
    
    v <- sapply(data, length)
    v <- v/sum(v)
    names(v) <- c("VOID", "PAWN", "ROOK", "KNIGHT", "BISHOP", "QUEEN", "KING")

    png(filename = paste0(graphs_wd, model, "/Distribution/distribution_", sprintf("%010d", number), ".png"), width = 700, height = 600)
    cols <- c("gray", rep("steelblue", 6))
    barplot(v, main = paste("Distribution", number), ylim = c(0, 1), col = cols)
    dev.off()
  }
  
  # Plot indicators #
  
  # Correct movements
  png(filename = paste0(graphs_wd, model, "/correct_movements.png"), width = 700, height = 600)
  plot(NA, ylim = c(0,1), ylab = "Frequency", bty = 'L', xlim = c(1,length(files)))
  
  for (j in 1:length(correct_frequencies)) {
    lines(correct_frequencies[[j]], col = j, pch = 16, type = 'o')
  }
  legend(1, 1, legend = names(correct_frequencies), col = 1:length(correct_frequencies), lty = 1, ncol = length(correct_frequencies))
  dev.off()
  
  # Correct destinations
  png(filename = paste0(graphs_wd, model, "/correct_destinations.png"), width = 700, height = 600)
  plot(NA, ylim = c(0,1), ylab = "Frequency", bty = 'L', xlim = c(1,length(files)))
  
  for (j in 1:length(correct_destinations)) {
    lines(correct_destinations[[j]], col = j, pch = 16, type = 'o')
  }
  legend(1, 1, legend = names(correct_destinations), col = 1:length(correct_destinations), lty = 1, ncol = length(correct_destinations))
  dev.off()
  
  setwd(base_wd)
  
  cat("\n")
}

cat("[R] DONE\n")
