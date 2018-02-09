# script to evaluate results of xing replication

library(ggplot2)
library(plyr)
library(dplyr)
library(magrittr)
library(forcats)
source('evaluate_xing_results_utils.R')
model_performace_summary = 'josh_gardner_xing-nibbler_model_performace_summary.csv'
results = read.csv(model_performace_summary)
results$model = factor(gsub('preds.', '', results$model))

#drop_courses = c("aidsfearandhope", "questionnairedesign")
#results %<>% filter(!(course %in% drop_courses))
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

#exploratory plots; not actually used but interesting to look at
ggplot(results, aes(x = week, y = auc, fill = model)) + geom_bar(stat = "identity", position = "dodge", alpha = 0.95) + facet_grid(feat_type ~ course) + scale_fill_brewer(palette = 1, direction = -1)
ggplot(results, aes(x = week, y = precision, fill = model)) + geom_bar(stat = "identity", position = "dodge", alpha = 0.95) + facet_grid(feat_type ~ course) + scale_fill_brewer(palette = 1, direction = -1)
ggplot(results, aes(x = week, y = auc, colour = model)) + geom_line() + facet_grid(feat_type ~ course) + scale_colour_manual(values=cbPalette) + theme_light() + ggtitle("AUC Stability By Course and Feature Type") + theme(plot.title = element_text(hjust = 0.5))
ggplot(results, aes(x = week, y = precision, colour = model)) + geom_line() + facet_grid(feat_type ~ course) + scale_colour_manual(values=cbPalette) + theme_light() + ggtitle("Precision Stability By Course and Feature Type") + theme(plot.title = element_text(hjust = 0.5))

# avg auc over time
ddply(results, .(week, feat_type, model), summarize, avg_auc = mean(auc, na.rm = T)) %>% ggplot(aes(x = week, y = avg_auc, colour = model)) + geom_line() + facet_grid(feat_type ~ .) + scale_colour_manual(values=cbPalette) + theme_light() + theme(plot.title = element_text(hjust = 0.5))
# auc with confidence bands 
conf.interval = .95
auc_plot = ddply(results, c("model", "feat_type", "week"), summarize, N = length(auc), mean_auc = mean(auc, na.rm = TRUE), sd = sd(auc, na.rm = TRUE), se = sd/sqrt(N), ci = qt(conf.interval/2 + .5, N-1)* se) %>%
    mutate(model = revalue(model, c("bn" = "BN", "ensemble" = "Ensemble", "tree" = "C4.5"))) %>% 
    ggplot(aes(x = week, y = mean_auc, colour = feat_type)) + geom_line(size = 1.5) + geom_errorbar(aes(ymin=mean_auc-ci, ymax=mean_auc+ci), width=.1, position = position_dodge(0.1)) + facet_grid(. ~ model) + scale_colour_manual(values=cbPalette) + theme_light() + theme(plot.title = element_text(hjust = 0.5)) + ylim(c(0,1)) + scale_x_continuous(breaks = seq(0,10), limits = c(0,10)) + labs(x = "Course Week", y="AUC", colour = "Feature \nType") + ggtitle("AUC Over Course Weeks") + theme(plot.title = element_text(hjust = 0.5), panel.grid.minor.x = element_blank())
ggsave("auc_plot.pdf", auc_plot, width = 12, height = 3.5, units = "in")
# precision with confidence bands
prec_plot = ddply(results, c("model", "feat_type", "week"), summarize, N = length(precision), mean_precision = mean(precision, na.rm = TRUE), sd = sd(precision, na.rm = TRUE), se = sd/sqrt(N), ci = qt(conf.interval/2 + .5, N-1)* se) %>%
    mutate(model = revalue(model, c("bn" = "BN", "ensemble" = "Ensemble", "tree" = "C4.5"))) %>% 
    ggplot(aes(x = week, y = mean_precision, colour = feat_type)) + geom_line(size = 1.5) + geom_errorbar(aes(ymin=mean_precision-ci, ymax=mean_precision+ci), width=.1, position = position_dodge(0.1)) + facet_grid(. ~ model) + scale_colour_manual(values=cbPalette) + theme_light() + theme(plot.title = element_text(hjust = 0.5)) + ylim(c(0,1)) + scale_x_continuous(breaks = seq(0,10), limits = c(0,10)) + labs(x = "Course Week", y="Precision", colour = "Feature \nType") + ggtitle("Precision Over Course Weeks") + theme(plot.title = element_text(hjust = 0.5), , panel.grid.minor.x = element_blank())
ggsave("prec_plot.pdf", prec_plot, width = 12, height = 3.5, units = "in")


# stat tests by model: compare ensemble to base learners
test_results_model <- data.frame()
for (mods in list(c("ensemble", "bn"), c("ensemble", "tree"))){
    for (ft in c("appended", "sum", "only")){
        mod1 = mods[1]
        mod2 = mods[2]
        z = fbh_auc_compare(results, compare_values = mods, compare_col = "model", filter_col = "feat_type", filter_value=ft)
        p = 2*pnorm(-abs(z))
        result_row = data.frame("mod1" = mod1, "mod2" = mod2, "feat_type" = ft, "z" = round(z,1), "p-value" = round(p, 4))
        test_results_model = rbind(test_results_model, result_row)
    }
}


# # compare feature types
# fbh_auc_compare(results, compare_values = c("appended", "only"), compare_col = "feat_type", filter_value = "tree", filter_col = "model")
# fbh_auc_compare(results, compare_values = c("appended", "sum"), compare_col = "feat_type", filter_value = "ensemble", filter_co = "model")
# fbh_auc_compare(results, compare_values = c("appended", "only"), compare_col = "feat_type", filter_value = "ensemble", filter_co = "model")

# stat tests by feature type: compare each set of models 
test_results_feat <- data.frame()
for (feats in list( c("appended", "only"), c("appended", "sum"), c("sum", "only"))){
    for (mod in c("tree", "bn", "ensemble")){
        feat1 = feats[1]
        feat2 = feats[2]
        z = fbh_auc_compare(results, compare_values = feats, compare_col = "feat_type", filter_col = "model", filter_value = mod)
        p = 2*pnorm(-abs(z))
        result_row = data.frame("feat1" = feat1, "feat2" = feat2, "model" = mod, "z" = round(z,1), "p-value" = round(p, 4))
        test_results_feat = rbind(test_results_feat, result_row)
        }
}

write.csv(test_results_model, file = "test_results_model.csv", row.names = FALSE)
write.csv(test_results_feat, file = "test_results_feat.csv", row.names = FALSE)
