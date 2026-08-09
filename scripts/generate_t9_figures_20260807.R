# T9 Nature-style figures from audited JSON/CSV artifacts.
library(ggplot2)
library(patchwork)
library(jsonlite)

root <- normalizePath(getwd(), mustWork = TRUE)
if (!dir.exists(file.path(root, "outputs")) && dir.exists(file.path(root, "..", "outputs"))) {
  root <- normalizePath(file.path(root, ".."), mustWork = TRUE)
}
path <- function(...) file.path(root, ...)
required <- c(
  path("outputs", "flair_shuffled_pairing_bootstrap_summary.json"),
  path("outputs", "semantic_drone_confirmation_bootstrap_3seed.json"),
  path("evidence", "coverage_matrix_20260805.csv"),
  path("evidence", "oem_seed_metrics_20260805.csv"),
  path("evidence", "oem_contrasts_20260805.csv"),
  path("evidence", "oem_leaf_contrasts_20260805.csv")
)
stopifnot(all(file.exists(required)))

palette <- c(
  "B0" = "#555555", "B0-half" = "#A8A8A8", "B1" = "#0072B2",
  "B1-shuffle" = "#D55E00", "positive" = "#009E73", "negative" = "#B33D3D",
  "exact" = "#0072B2", "absent" = "#F5D0C8", "ignored" = "#D9D9D9",
  "not_scored" = "#F5F5F5", "LoveDA" = "#555555",
  "native FLAIR" = "#0072B2", "deranged FLAIR" = "#D55E00", "none" = "#F5F5F5"
)

theme_paper <- function(base_size = 7) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.5, colour = "#2B2B2B"),
      legend.title = element_blank(), legend.text = element_text(size = base_size - 0.7),
      plot.title = element_text(size = base_size + 1, face = "bold", hjust = 0),
      plot.tag = element_text(size = base_size + 1.2, face = "bold"),
      strip.background = element_blank(), strip.text = element_text(size = base_size, face = "bold"),
      panel.grid = element_blank(), plot.margin = margin(4, 5, 4, 5)
    )
}

save_figure <- function(plot, stem, width, height) {
  grDevices::cairo_pdf(path("figures", paste0(stem, ".pdf")), width = width, height = height, family = "Arial")
  print(plot)
  dev.off()
  svglite::svglite(path("figures", paste0(stem, ".svg")), width = width, height = height); print(plot); dev.off()
  ragg::agg_png(path("figures", paste0(stem, ".png")), width = width, height = height, units = "in", res = 300); print(plot); dev.off()
  ragg::agg_tiff(path("figures", paste0(stem, ".tiff")), width = width, height = height, units = "in", res = 600); print(plot); dev.off()
}

shuffle <- fromJSON(path("outputs", "flair_shuffled_pairing_bootstrap_summary_v2.json"), simplifyVector = FALSE)
semantic_drone <- fromJSON(path("outputs", "semantic_drone_confirmation_bootstrap_3seed.json"), simplifyVector = FALSE)
seed_metrics <- read.csv(path("evidence", "oem_seed_metrics_20260805.csv"), stringsAsFactors = FALSE)
contrasts <- read.csv(path("evidence", "oem_contrasts_20260805.csv"), stringsAsFactors = FALSE, na.strings = "NA")
leaf_primary <- read.csv(path("evidence", "oem_leaf_contrasts_20260805.csv"), stringsAsFactors = FALSE)
shuffle_seeds <- do.call(rbind, lapply(shuffle$arms$b1_shuffle$per_seed, function(x) {
  data.frame(arm = "B1-shuffle", seed = x$seed, mean_iou = x$mean_iou,
             rangeland_iou = x$per_class_iou$herbaceous_vegetation)
}))
seed_plot <- rbind(seed_metrics[, c("arm", "seed", "mean_iou", "rangeland_iou")], shuffle_seeds)
seed_plot$arm <- factor(seed_plot$arm, levels = c("B0", "B0-half", "B1-shuffle", "B1"))

# Fig. 1 contract: label support -> source pairing -> target contrasts.
coverage <- read.csv(path("evidence", "coverage_matrix_20260805.csv"), stringsAsFactors = FALSE)
coverage$leaf <- factor(coverage$leaf, levels = c("built_structure", "transport_surface", "woody_vegetation", "herbaceous_vegetation", "cropland", "bare_surface", "surface_water"))
coverage$dataset <- factor(coverage$dataset, levels = rev(c("LoveDA", "FLAIR", "OpenEarthMap", "Semantic Drone")))
coverage$support_status <- factor(coverage$support_status, levels = c("exact", "absent", "ignored", "not_scored"))
coverage$label <- ifelse(coverage$support_status == "exact", "exact", ifelse(coverage$support_status == "absent", "absent", ""))
coverage$text_color <- ifelse(coverage$support_status == "exact", "white", "#263238")
p1a <- ggplot(coverage, aes(leaf, dataset, fill = support_status)) +
  geom_tile(colour = "white", linewidth = 0.65, height = 0.84) +
  geom_tile(data = subset(coverage, support_status == "absent"), fill = NA,
            colour = "#B23A2B", linewidth = 0.9, height = 0.84) +
  geom_text(aes(label = label, colour = text_color), size = 2.45, fontface = "bold") +
  scale_colour_identity() +
  scale_fill_manual(values = palette[c("exact", "absent", "ignored", "not_scored")],
                    labels = c("Exact support", "Absent", "Ignored", "Not scored"),
                    name = NULL, drop = FALSE) +
  scale_x_discrete(labels = c("Built", "Transport", "Woody", "Herbaceous", "Crop-\nland", "Bare", "Water")) +
  labs(title = "Label-support audit", x = NULL, y = NULL) + theme_paper(base_size = 7.3) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, size = 6.7),
        axis.text.y = element_text(size = 6.8), axis.ticks = element_blank(),
        legend.position = "bottom", legend.direction = "horizontal",
        legend.key.height = unit(3.2, "mm"), legend.key.width = unit(8.5, "mm"),
        legend.text = element_text(size = 6.1), plot.margin = margin(4, 7, 2, 5))

controls <- data.frame(
  arm = rep(c("B0", "B0-half", "B1", "B1-shuffle"), each = 2), slot = rep(c("Slot 1", "Slot 2"), 4),
  source = c("LoveDA", "LoveDA", "LoveDA", "none", "LoveDA", "native FLAIR", "LoveDA", "deranged FLAIR"),
  label = c("LoveDA", "LoveDA", "LoveDA", "", "LoveDA", "FLAIR RGB +\nnative ID-10 mask", "LoveDA", "FLAIR RGB +\nderanged ID-10 mask")
)
controls$arm <- factor(controls$arm, levels = rev(c("B0", "B0-half", "B1", "B1-shuffle")))
controls$slot <- factor(controls$slot, levels = c("Slot 1", "Slot 2"))
controls$source <- factor(controls$source, levels = c("LoveDA", "native FLAIR", "deranged FLAIR", "none"))
p1b <- ggplot(controls, aes(slot, arm, fill = source)) +
  geom_tile(colour = "white", linewidth = 0.85, height = 0.72) +
  geom_text(aes(label = label), size = 2.55, lineheight = 0.88,
            colour = ifelse(controls$source %in% c("LoveDA", "native FLAIR", "deranged FLAIR"), "white", "#2B2B2B")) +
  scale_fill_manual(values = palette[c("LoveDA", "native FLAIR", "deranged FLAIR", "none")]) +
  scale_x_discrete(labels = c("LoveDA\nslot 1", "Source\nslot 2")) +
  labs(title = "Fixed-budget pairing controls",
       subtitle = "25,220 fixed updates; B1 and B1-shuffle share source exposure",
       x = NULL, y = NULL) + theme_paper(base_size = 7.3) +
  theme(legend.position = "none", axis.ticks = element_blank(),
        axis.text.x = element_text(size = 6.8), axis.text.y = element_text(size = 6.8),
        plot.subtitle = element_text(size = 6.1, colour = "#5B6770", hjust = 0),
        plot.margin = margin(3, 7, 2, 5))

b1_b0 <- contrasts[contrasts$contrast == "B1-B0" & contrasts$metric == "rangeland_iou", ]
b1_shuffle <- shuffle$contrasts$b1_minus_b1_shuffle$overall
sd_b1_b0 <- semantic_drone$comparisons$paired_raster_bootstrap_b1_minus_b0
headlines <- data.frame(
  comparison = factor(c("OEM: B1 - B0", "OEM: B1 - B1-shuffle", "Semantic Drone: B1 - B0"),
                      levels = rev(c("OEM: B1 - B0", "OEM: B1 - B1-shuffle", "Semantic Drone: B1 - B0"))),
  estimate = c(b1_b0$estimate, b1_shuffle$point_difference$per_class_iou$herbaceous_vegetation, sd_b1_b0$point_difference),
  lower = c(b1_b0$ci_lower, b1_shuffle$confidence_interval$per_class_iou$herbaceous_vegetation$lower, sd_b1_b0$confidence_interval$lower),
  upper = c(b1_b0$ci_upper, b1_shuffle$confidence_interval$per_class_iou$herbaceous_vegetation$upper, sd_b1_b0$confidence_interval$upper)
)
headlines$label_x <- pmin(headlines$upper + 0.009, 0.365)
p1c <- ggplot(headlines, aes(estimate, comparison)) +
  geom_vline(xintercept = 0, linewidth = 0.4, colour = "#4D4D4D") +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = 0.13,
                linewidth = 0.9, colour = palette["positive"]) +
  geom_point(size = 2.9, colour = palette["positive"]) +
  geom_text(aes(x = label_x, label = sprintf("%+.3f", estimate)), hjust = 0,
            size = 2.65, colour = "#263238", fontface = "bold") +
  coord_cartesian(xlim = c(-0.03, 0.38), clip = "off") +
  labs(title = "Observed target-leaf contrasts",
       subtitle = "95% paired bootstrap intervals; estimates conditional on fixed checkpoints",
       x = "Paired target-leaf IoU difference", y = NULL) + theme_paper(base_size = 7.3) +
  theme(axis.text.y = element_text(size = 6.8),
        plot.subtitle = element_text(size = 6.1, colour = "#5B6770", hjust = 0),
        plot.margin = margin(3, 18, 5, 5))
fig1 <- (p1a / p1b / p1c) +
  plot_layout(heights = c(2.45, 1.85, 1.65)) +
  plot_annotation(tag_levels = "a", theme = theme(plot.tag = element_text(size = 9, face = "bold", colour = "#263238")))
save_figure(fig1, "fig1_research_design_20260807", 7.2, 6.35)

# Fig. 2 contract: show checkpoint-level target-leaf values and compact paired contrasts.
set.seed(20260807)
p2a <- ggplot(seed_plot, aes(arm, rangeland_iou, colour = arm)) +
  geom_jitter(width = 0.08, height = 0, size = 2.2) +
  stat_summary(fun = mean, geom = "crossbar", width = 0.48, colour = "black", linewidth = 0.45) +
  scale_colour_manual(values = palette[c("B0", "B0-half", "B1-shuffle", "B1")]) +
  coord_cartesian(ylim = c(-0.015, 0.38)) +
  annotate("text", x = 3.45, y = 0.365, label = "B1 - B0: +0.312 [0.282, 0.340]\nB1 - B1-shuffle: +0.039 [0.033, 0.044]", size = 1.80, hjust = 0.5) +
  labs(x = NULL, y = "Rangeland IoU") + theme_paper() + theme(legend.position = "none")

b1b0_rows <- contrasts[contrasts$contrast == "B1-B0" & contrasts$metric %in% c("rangeland_iou", "mean_iou", "supported_leaf_miou"), ]
b1b0_rows$comparison <- "B1 - B0"
b1shuffle_rows <- data.frame(
  comparison = "B1 - B1-shuffle", metric = c("rangeland_iou", "mean_iou", "supported_leaf_miou"),
  estimate = c(b1_shuffle$point_difference$per_class_iou$herbaceous_vegetation, b1_shuffle$point_difference$mean_iou, shuffle$contrasts$b1_minus_b1_shuffle$supported_leaf$point_difference),
  ci_lower = c(b1_shuffle$confidence_interval$per_class_iou$herbaceous_vegetation$lower, b1_shuffle$confidence_interval$mean_iou$lower, shuffle$contrasts$b1_minus_b1_shuffle$supported_leaf$confidence_interval$lower),
  ci_upper = c(b1_shuffle$confidence_interval$per_class_iou$herbaceous_vegetation$upper, b1_shuffle$confidence_interval$mean_iou$upper, shuffle$contrasts$b1_minus_b1_shuffle$supported_leaf$confidence_interval$upper)
)
summary_rows <- rbind(b1b0_rows[, c("comparison", "metric", "estimate", "ci_lower", "ci_upper")], b1shuffle_rows)
summary_rows$metric <- factor(summary_rows$metric, levels = rev(c("rangeland_iou", "mean_iou", "supported_leaf_miou")), labels = c("Rangeland IoU", "Overall mIoU", "Supported-leaf mIoU"))
summary_rows$sign <- ifelse(summary_rows$estimate >= 0, "positive", "negative")
p2b <- ggplot(summary_rows, aes(estimate, metric, colour = sign)) +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = "#777777") +
  geom_errorbar(data = subset(summary_rows, !is.na(ci_lower)), aes(xmin = ci_lower, xmax = ci_upper), orientation = "y", width = 0.13, linewidth = 0.65) +
  geom_point(size = 2.25) + facet_wrap(~comparison, nrow = 1) +
  scale_colour_manual(values = palette[c("positive", "negative")], guide = "none") +
  labs(x = "Paired city-bootstrap difference", y = NULL) + theme_paper()
save_figure(p2a + p2b + plot_layout(widths = c(1.05, 1.55)) + plot_annotation(tag_levels = "a"), "fig2_oem_pairing_control_20260807", 7.2, 3.2)

# Fig. 3 contract: separate the native-vs-baseline trade-off from the pairing contrast.
leaf_order <- c("built_structure", "transport_surface", "woody_vegetation", "herbaceous_vegetation", "cropland", "bare_surface", "surface_water")
leaf_b1_b0 <- leaf_primary[leaf_primary$contrast == "B1-B0", c("leaf", "estimate", "ci_lower", "ci_upper")]
leaf_b1_b0$comparison <- "B1 - B0"
shuffle_leaf <- b1_shuffle$point_difference$per_class_iou
shuffle_ci <- b1_shuffle$confidence_interval$per_class_iou
leaf_b1_shuffle <- do.call(rbind, lapply(leaf_order, function(leaf) {
  data.frame(leaf = leaf, estimate = shuffle_leaf[[leaf]], ci_lower = shuffle_ci[[leaf]]$lower, ci_upper = shuffle_ci[[leaf]]$upper)
}))
leaf_b1_shuffle$comparison <- "B1 - B1-shuffle"
leaf_plot <- rbind(leaf_b1_b0, leaf_b1_shuffle)
leaf_labels <- c("built_structure" = "Built structure", "transport_surface" = "Transport surface", "woody_vegetation" = "Woody vegetation", "herbaceous_vegetation" = "Herbaceous vegetation", "cropland" = "Cropland", "bare_surface" = "Bare surface", "surface_water" = "Surface water")
leaf_plot$leaf <- factor(leaf_plot$leaf, levels = rev(leaf_order), labels = rev(unname(leaf_labels[leaf_order])))
leaf_plot$sign <- ifelse(leaf_plot$estimate >= 0, "positive", "negative")
p3 <- ggplot(leaf_plot, aes(estimate, leaf, colour = sign)) +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = "#777777") +
  geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper), orientation = "y", width = 0.13, linewidth = 0.6) +
  geom_point(size = 2.0) + facet_wrap(~comparison, nrow = 1) +
  scale_colour_manual(values = palette[c("positive", "negative")], guide = "none") +
  coord_cartesian(xlim = c(-0.30, 0.36)) + labs(x = "Per-leaf IoU difference (paired city 95% CI)", y = NULL) + theme_paper()
save_figure(p3, "fig3_leafwise_pairing_contrasts_20260807", 7.2, 3.7)
