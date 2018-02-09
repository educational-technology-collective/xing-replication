library(dplyr)
library(tidyr)

# ## this is incomplete -- see notes
# sign_test_by_feat_type <- function(results_df, outcome = "auc") {
#     feat_types = unique(results_df$feat_type)
#     for (ft in feat_types){
#         temp = filter(results_df, feat_type == ft) %>% 
#             select(one_of(c("model", outcome, "week", "course"))) %>% 
#             spread(model, auc)
#         temp$sign_ensemble_vs_bn = (temp$ensemble > temp$bn)
#         temp$sign_ensemble_vs_tree = (temp$ensemble > temp$tree)
#         # todo: distribute ties evenly
#         p = sign_test(N = (sum(temp$sign_ensemble_vs_bn) + 0.5*sum(temp$ensemble == temp$bn)))
#         print(dim(temp))
#     }
# }

## compute standard error of AUC score, using its equivalence to the Wilcoxon statistic (See Fogarty, Baker and Hudson, "Case Studies in the use of ROC Curve Analysis for Sensor-Based Estimates in Human Computer Interaction" 2008.)
## auc: value of A' statistic.
## n_p: number of positive cases.
## n_n: number of negative cases.
se_auc <- function(auc, n_p, n_n){
    D_p = (n_p - 1)*( (auc/(2 - auc)) - auc^2)
    D_n = (n_n - 1)*((2 * auc^2)/(1+auc) - auc^2) 
    SE_auc = sqrt( (auc * (1-auc) + D_p + D_n) / (n_p * n_n) )
    return(SE_auc)
}

## apply z-test for difference between auc_1 and auc_2 using FBH method.
fbh_test <- function(auc_1, auc_2, n_p, n_n){
    SE_auc_1 = se_auc(auc_1, n_p, n_n)
    SE_auc_2 = se_auc(auc_2, n_p, n_n)
    z = (auc_1 - auc_2)/sqrt(SE_auc_1^2 + SE_auc_2^2)
    return(z)
}

## compute aggregate z-score using stouffer's method, ignoring NA values
## z_vec: vector of z-scores.
stouffer_z <- function(z_vec){
    S_z = sum(z_vec, na.rm=TRUE)
    k = sum(!is.na(z_vec))
    stouffers_z = S_z/sqrt(k)
    return(stouffers_z)
}

## function to apply FBH method to compare outcome_col by compare_col, averaging over time_col (due to non-independence) and then over over_col.
## See Fogarty, Baker and Hudson, "Case Studies in the use of ROC Curve Analysis for Sensor-Based Estimates in Human Computer Interaction" 2008.
##
## df: DataFrame containing time_col, outcome_col, compare_col, and over_col.
## compare_values: names of models to compare (vector of length 2). These should match exactly their names in compare_col.
## time_col: name of column in df representing time of observations (z-scores are averaged over time_col within each model/course due to non-independence).
## outcome_col: name of column in df representing outcome to compare; note that this method applies specifically to AUC and not other metrics (i.e., sensitivity, precision, F1).
## compare_col: name of column in df representing two conditions to compare (should contain at least 2 unique values).
## over_col: different experiments over which z-scores for models are to be compared (using Stouffer's Z; see  Stouffer, S.A.; Suchman, E.A.; DeVinney, L.C.; Star, S.A.; Williams, R.M. Jr. (1949). The American Soldier, Vol.1: Adjustment during Army Life.)
## n_col: name of column in df with total N.
## n_p_col: name of column in df with n_p, number of positive observations.
## n_n_col: name of column in df with n_n, number of negative observations.
## filter_col: name of column in df with feature type (if these exist).
fbh_auc_compare <- function(df, compare_values, filter_value, time_col = "week", outcome_col = "auc", compare_col = "model", over_col = "course", n_col = "n", n_p_col = "n_p", n_n_col = "n_n", filter_col = "feat_type"){
    filter_str = paste0(compare_col, " %in% c('", compare_values[1], "', '", compare_values[2], "') & ", filter_col, " == '", filter_value, "'")
    comp_df = select(df, one_of(c(time_col, outcome_col, compare_col, over_col, n_col, n_p_col, n_n_col, filter_col))) %>% dplyr::filter_(filter_str)
    n_courses = length(unique(comp_df[,over_col]))
    course_z_scores = rep(NA, n_courses)
    for (course_ix in seq_along(unique(comp_df[,over_col]))){
        course = unique(comp_df[,over_col])[course_ix]
        message(paste0("fetching comparison results for models ", compare_values[1], ", ", compare_values[2], " in course ", course, " with feature type ", filter_value))
        course_df = comp_df[comp_df[,over_col] == course,]
        n_weeks = length(unique(course_df[,time_col]))
        course_week_z_scores = rep(NA, n_weeks)
        for (week in unique(course_df[,time_col])){
            course_week_df = course_df[course_df[,time_col] == week,]
            # check whether both models have valid output for this week; otherwise skip
            if(length(unique(course_week_df[,compare_col])) != length(compare_values)){
                msg = paste0("Missing performance data for at least one model in course ", course, " week ", week, "; skipping")
                message(msg)
                next
            }
            auc.1 = course_week_df[course_week_df[,compare_col] == compare_values[1], outcome_col]
            auc.2 = course_week_df[course_week_df[,compare_col] == compare_values[2], outcome_col]
            n.p = course_week_df[1,n_p_col]
            n.n = course_week_df[1,n_n_col]
            # conduct test and store z score in course_week_z_scores
            z = fbh_test(auc_1 = auc.1, auc_2 = auc.2, n_p = n.p, n_n = n.n)
            course_week_z_scores[week] <- z
        }
        # calculate average z-score across weeks and store
        course_z = mean(course_week_z_scores, na.rm = TRUE)
        course_z_scores[course_ix] <- course_z 
    }
    # applying stouffer's Z method to course_z_scores
    overall_z = stouffer_z(course_z_scores)
    return(overall_z)
}

