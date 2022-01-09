library(kde1d)

func_draw_unconstrained_kde <- function(percentages){
  # Draw unconstrained KDE
  #
  # Args:
  #   percentages: vector of percentages.
  #
  # Returns:
  #   KDE.
  fit <- kde1d(percentages) # estimate density
  summary(fit)              # information about the estimate
  plot(fit, lwd=0.5, main = "Unconstrained KDE", xlab = "Percentages") # plot the density estimate

  return(fit)

}

func_get_prob_mass_trans <- function(fit_bounded, a, b){
  # get the area under the curve in a given range.
  #
  # Args:
  #   fit_bounded: KDE.
  #   a: lower value.
  #   b: upper value.
  #
  # Returns:
  #   area under the curve between a and b.
  auc_val <- pkde1d(b,fit_bounded) - pkde1d(a,fit_bounded)

  #print(paste("AUC from",a,"to",b,":",auc_val))

  return(auc_val)
}


func_modify_data <- function(percentages, low_cut, up_cut,low,up){
  # cut of extreme data at ends (if any)
  #
  # Args:
  #   percentages: unmodified percentages
  #   low_cut: where to cut if no peak at lower extreme
  #   up_cut: where to cut if no peak at upper extreme
  #   low: lower bound.
  #   up: upper bound.
  #
  # Returns:
  #   percentages after removing extremes.

  tot_data <- length(percentages)

  count_0 <- sum(abs(percentages - low) < 1e-6)
  count_100 <- sum(abs(percentages - up) < 1e-6)

  print(paste("Data percentage at",low,count_0/tot_data*100))
  print(paste("Data percentage at",up,count_100/tot_data*100))

  # cut off peaks at 0
  if(count_0/tot_data > 0.01){
    next_min <- min( percentages[percentages!=min(percentages)] )
    percentages <- subset(percentages, percentages>=next_min)
    print(paste('Cutting values below',next_min))
    print(length(percentages))

  # no peak, cut off at given low_cut
  }else{
    percentages <- subset(percentages, percentages>=low_cut)
    print(paste('Cutting values below',low_cut))
    print(length(percentages))
  }

  # cut off peaks at 100
  if(count_100/tot_data > 0.01){
    next_max <- max( percentages[percentages!=max(percentages)] )
    percentages <- subset(percentages, percentages<=next_max)
    print(paste('Cutting values above',next_max))
    print(length(percentages))

  # no peak, cut off at given up_cut
  }else{
    percentages <- subset(percentages, percentages<=up_cut)
    print(paste('Cutting values above',up_cut))
    print(length(percentages))
  }

  return(percentages)

}

func_cut_off_clusters <- function(raw_percentages, low_cut, up_cut, low=0, up=100){
  # draws new KDE after removing extreme points.
  #
  # Args:
  #   raw_percentages: unmodified percentages
  #   low_cut: where to cut if no peak at lower extreme
  #   up_cut: where to cut if no peak at upper extreme
  #   low: lower bound.
  #   up: upper bound.
  #
  # Returns:
  #   modified KDE.
  fit <- func_draw_unconstrained_kde(raw_percentages)
  percentages <- func_modify_data(raw_percentages, low_cut, up_cut, low, up)
  new_fit <- func_draw_unconstrained_kde(percentages)
  lower_bound <- floor(qkde1d(0,new_fit))
  upper_bound <- ceiling(qkde1d(1,new_fit))
  print(paste('Lower bound', lower_bound))
  print(paste('Upper bound', upper_bound))
  print(paste('AUC between lower bound and',low, pkde1d(low,new_fit)))
  print(paste('AUC between upper bound and',up, pkde1d(upper_bound,new_fit) - pkde1d(up,new_fit)))
  print("Unconstrained KDE after cutting off peaks")
  return(new_fit)

}

func_plot_final_kde <- function(fit, name){
  # Plot final KDE
  #
  # Args:
  #   fit: final KDE.
  #   name: plot name.

  jpeg(paste('graphs/',name,'_KDE.jpg', sep=""))
  plot(fit, lwd=0.5, main = paste(name," KDE"), xlab = "Percentages") # plot the density estimate
  dev.off()

}

func_plot_new_kde <- function(fit, name){
  # Plot final KDE
  #
  # Args:
  #   fit: final KDE.
  #   name: plot name.

  jpeg(paste('graphs/',name,'_KDE.jpg', sep=""))
  plot(fit, lwd=0.5, main = paste(name," KDE"), xlab = "Norm") # plot the density estimate
  dev.off()

}

func_get_all_kdes <- function(rule_type, bounds, percentages, rule_names){
  # Gets all KDEs

  t <- 0.5
  num_rules <- length(percentages)
  kdes <- vector("list", num_rules)

  for (i in 1:num_rules){
      if (rule_type[[i]]=='lower'){
          print(rule_names[[i]])
          kde_fit <- func_cut_off_clusters(unlist(percentages[[i]]),0+t,100-t)
          kdes[[i]] <- kde_fit
          func_plot_new_kde(kde_fit, rule_names[[i]])
      } else if (rule_type[[i]]=='higher'){
          print(rule_names[[i]])
          kde_fit <- func_cut_off_clusters(unlist(percentages[[i]]),0+t,100-t)
          kdes[[i]] <- kde_fit
          func_plot_new_kde(kde_fit, rule_names[[i]])
      } else if (rule_type[[i]]=='mid'){
          print(rule_names[[i]])
          p <- unlist(percentages[[i]])
          s <- bounds[[i]][[1]]
          e <- bounds[[i]][[2]]
          sub1 <- subset(p, p <= s)
          sub2 <- subset(p, p >= e)
          sub1_fit <- func_cut_off_clusters(sub1,0+t,s-t,0,s)
          func_plot_new_kde(sub1_fit, paste(rule_names[[i]],"_g1"))
          print(paste("test", length(sub2)))
          sub2_fit <- func_cut_off_clusters(sub2,e+t,100-t,e,100)
          func_plot_new_kde(sub2_fit, paste(rule_names[[i]],"_g2"))
          kdes[[i]] <- list(sub1_fit, sub2_fit)
      }

  }
  return(kdes)
}


