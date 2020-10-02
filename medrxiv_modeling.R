#########
#Spline fitting used for Fig. 6. Deviations from typical healthy behavior and physiology observed during ILI events.
# Generalized additive models were fit to the excess values separately for each cohort (mgcv package for R) and the resulting regression splines were used for visualizations

install.packages("mgcv")
library(mgcv)
mgcv::gam(y_resid ~ s(days_since_symptoms_onset) + s(user_id, bs="re"), data=resid_data, method = 'REML')

#where
# user_id = a 1-dimensional array of 1,352 user ids (across the three cohorts)
#
# y_resid = the residuals from the mixed effects model, *fit only to baseline days (all observation days=189 days, minus day -10..20 = 158 days)*:

install.packages("lme4")
library(lme4)
lme4::lmer(y ~ week_L + week_Q + week_C + mean_y_in_state_on_date + day_of_week + (1 | user_id), data = baseline_period_data)
#where in baseline_period_data we’re working with vectors of length N participants * M baseline observations. In our case we have 1,352 participants * 158 baseline observation days.
#week_L, week_Q, and week_C are the 1st, 2nd, and 3rd order expansions of the ordinal variable week, week_L = 1 for the first week of the study

#To get the residual you predict activity on all days we perform the two steps below:

#steps_bl__predicted: the mixed effects model is estimated on baseline_period_data and then used to predict behavior on all_data - we are making N=1,352 participants * M=189, one for each row (participant-day) in all_data  with no missing values
all_data$steps_bl__predicted <- predict(STEPS_model_baseline, all_data, allow.new.levels=T)

#steps__bl_resids: Then we subtract the observed behavior values from the predicted behavior to get the excess: steps_bl__predicted -  steps__observed, but there will be NA values when steps__observed is NA
all_data$steps__bl_resids <- all_data$steps__observed - all_data$steps_bl__predicted


#########
# Normalizing RHR is z-scoring RHR, used in Fig. 5:

function(x, na.rm = FALSE) (x - mean(x, na.rm = T)) / sd(x, na.rm = T)
