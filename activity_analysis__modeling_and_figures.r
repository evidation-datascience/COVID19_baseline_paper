# Databricks notebook source
# MAGIC %md #Activity Analysis: Modeling and Figures

# COMMAND ----------

# MAGIC %md #### Load dependencies

# COMMAND ----------

library(SparkR)
library(dplyr)
library(ggplot2)
library(stringr)
library(tidyr)
library(purrr)
library(forcats)

install.packages("mgcv")
library(mgcv)

install.packages("lme4")
library(lme4)

options(scipen=999)

# COMMAND ----------

# Define Figure Formats

theme_basic <- theme(
  axis.text = element_text(size=27, colour="#3d3d3d", family="sans"),
  axis.title.x = element_text(colour="#636363", family="sans", size = 32, margin = margin(t = 20, r = 0, b = 0, l = 0)),
  axis.title.y = element_text(colour="#636363", family="sans", size = 27),
  legend.title = element_text(colour="#3d3d3d", family="sans", face="bold"),
  legend.text =  element_text(colour="#3d3d3d", family="sans"),
  plot.title =element_text(size = 27, colour="#3d3d3d", family="sans", face="bold", hjust = 0.5, vjust=2),
  plot.subtitle = element_text(colour="#3d3d3d", family="sans", hjust = 0.5, vjust=2 ),
  strip.background = element_rect(color="white", fill="white", size=1.5, linetype="solid"),
  strip.text = element_text(colour="black", family="sans", size = 27),
  strip.text.y.left = element_text(angle=0),
  strip.placement = "outside",
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.background = element_rect(fill="white"),
  panel.spacing = unit(5, "lines"))

theme_density <- theme(
  axis.text = element_text(size=24, colour="#3d3d3d", family="sans"),
  axis.text.x = element_text(size=24, angle = 45, hjust=1),
  axis.title = element_text(size=24, colour="#303030", family="sans"),
  axis.title.x = element_blank(),
  legend.title = element_blank(),
  legend.text =  element_text(colour="#3d3d3d", family="sans", size = 24),
  legend.key = element_rect(color="black"),
  strip.background = element_rect(color="white", fill="white", linetype="solid"),
  strip.text = element_text(colour="black", family="sans", size = 24),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.background = element_rect(fill="white"))
  
                     # covid    # Non-Covid Flu  # Pre-Covid  
cohort_palette <- c("#287c85", "#787878", "#b8b8b8") 

# put the elements in a list
apply_theme_basic <- function(manual_pal){
  list(theme_basic, scale_fill_manual(values = manual_pal), scale_color_manual(values = manual_pal), guides(colour = F, fill = F))}

apply_theme_density <- function(manual_pal){
  list(theme_density, scale_fill_manual(values = manual_pal), scale_color_manual(values = manual_pal))}


# COMMAND ----------

# MAGIC %md ## Load activity data
# MAGIC 
# MAGIC Qualified researchers may download this dataset from Synapse: https://www.synapse.org/20192020ilisurveillanceprogram 

# COMMAND ----------

#NOTE: UPDATE THIS CELL WITH THE APPROPRIATE PATHNAMES

#define pathname of dataset
activity_data_path <- '/dbfs/data/covid19/manuscript_data_delivery_for_synapse/activity_data.csv.gz'

# COMMAND ----------

analysis_panel <- read.csv(activity_data_path)

analysis_panel %>% head() %>% display()

# COMMAND ----------

# verify that there are no unexpected values (output of cell should = 0 rows)
print(analysis_panel %>% 
        filter((valid_user__steps == 0 & !is.na(steps__observed)) | 
               (valid_day__steps == 0 & !is.na(steps__observed)) |
               (valid_day__steps == 1 & is.na(steps__observed)) |
               (valid_user__heart == 0 & !is.na(heart__observed)) |
               (valid_day__heart == 0 & !is.na(heart__observed)) |
               (valid_day__heart == 1 & is.na(heart__observed)) | 
               (valid_user__sleep == 0 & !is.na(sleep__observed)) |
               (valid_day__sleep == 0 & !is.na(sleep__observed)) |
               (valid_day__sleep == 1 & is.na(sleep__observed))
              ))


# COMMAND ----------

# MAGIC %md Print counts of users with valid steps, heart, and sleep data in each cohort

# COMMAND ----------

print(analysis_panel %>% select(user_id, cohort, starts_with("valid_user")) %>% unique() %>% group_by(cohort, valid_user__steps) %>% filter(valid_user__steps == 1) %>% droplevels() %>% tally())
print(analysis_panel %>% select(user_id, cohort, starts_with("valid_user")) %>% unique() %>% group_by(cohort, valid_user__heart) %>% filter(valid_user__heart == 1) %>% droplevels() %>% tally())
print(analysis_panel %>% select(user_id, cohort, starts_with("valid_user")) %>% unique() %>% group_by(cohort, valid_user__sleep) %>% filter(valid_user__sleep == 1) %>% droplevels() %>% tally())

# COMMAND ----------

# MAGIC %md
# MAGIC # Fit Model
# MAGIC ###  [activity feature] ~ week_L + week_Q + week_C + [mean_activity_in_state] + day_of_week (1 | user_id)
# MAGIC - 1st, 2nd, and 3rd extensions of week of year, categorical term for day of week
# MAGIC - random intercept for user's baseline
# MAGIC - dependent variable is winsorized activity value
# MAGIC - for state-wide mean activity data values, use scaled versions for steps and sleep to allow similar scales as other predictors, helps model converge

# COMMAND ----------

# Fit 1st set of models to ALL participant-days
STEPS_model <- lmer(steps__observed ~ week_L + week_Q + week_C + state__steps__scale1k + day_of_week + (1 | user_id), data = analysis_panel, na.action = na.exclude)
HEART_model <- lmer(heart__observed ~ week_L + week_Q + week_C + state__heart + day_of_week + (1 | user_id), data = analysis_panel, na.action = na.exclude)
SLEEP_model <- lmer(sleep__observed ~ week_L + week_Q + week_C + state__sleep__scale60 + day_of_week + (1 | user_id), data = analysis_panel, na.action = na.exclude)

# Isolate baseline days for the second set of models
analysis_panel_baseline <- analysis_panel %>% filter(flu_period=="baseline") %>% droplevels()

# Fit 2nd set of models to BASELINE participant-days only
STEPS_model_baseline <- lmer(steps__observed ~ week_L + week_Q + week_C + state__steps__scale1k + day_of_week + (1 | user_id), data = analysis_panel_baseline, na.action = na.exclude)
HEART_model_baseline <- lmer(heart__observed ~ week_L + week_Q + week_C + state__heart + day_of_week + (1 | user_id), data = analysis_panel_baseline, na.action = na.exclude)
SLEEP_model_baseline <- lmer(sleep__observed ~ week_L + week_Q + week_C + state__sleep__scale60 + day_of_week + (1 | user_id), data = analysis_panel_baseline, na.action = na.exclude)

print(summary(STEPS_model))
print(summary(HEART_model))
print(summary(SLEEP_model))
print(summary(STEPS_model_baseline))
print(summary(HEART_model_baseline))
print(summary(SLEEP_model_baseline))

# COMMAND ----------

# MAGIC %md # USE MODEL-FITTED ESTIMATES TO PREDICT MISSING VALUES

# COMMAND ----------

scale2 <- function(x, na.rm = FALSE) (x - mean(x, na.rm = T)) / sd(x, na.rm = T)

model_outputs <- analysis_panel

# Use model fit to all days to predict a value for every participant-day, these will be used as imputed values in elevated HR analysis
model_outputs$steps__predicted <- predict(STEPS_model, model_outputs, allow.new.levels=T)
model_outputs$heart__predicted <- predict(HEART_model, model_outputs, allow.new.levels=T)
model_outputs$sleep__predicted <- predict(SLEEP_model, model_outputs, allow.new.levels=T)

# Use model fit to only baseline days days to predict a value for every participant-day, these will be used to derive excess activity observed during ILI event
model_outputs$steps_bl__predicted <- predict(STEPS_model_baseline, model_outputs, allow.new.levels=T)
model_outputs$heart_bl__predicted <- predict(HEART_model_baseline, model_outputs, allow.new.levels=T)
model_outputs$sleep_bl__predicted <- predict(SLEEP_model_baseline, model_outputs, allow.new.levels=T)


model_outputs <- model_outputs %>%

  mutate( # Impute missing activity data values by filling NAs with predicted values from model fit to ALL days, these will be used as imputed values in elevated HR analysis
         steps__imputed = ifelse(valid_user__steps == 1 & is.na(steps__observed), steps__predicted,
                                 ifelse(valid_user__steps ==1, steps__observed, NA)),
         heart__imputed = ifelse(valid_user__heart == 1 & is.na(heart__observed), heart__predicted,
                                 ifelse(valid_user__heart ==1, heart__observed, NA)),
         sleep__imputed = ifelse(valid_user__sleep == 1 & is.na(sleep__observed), sleep__predicted,
                                 ifelse(valid_user__sleep ==1, sleep__observed, NA)),
         
        # Compute excess (residuals) associated with ILI events by subtracting the value predicted by the BASELINE model from the observed value
         steps__bl_resids =   steps__observed - steps_bl__predicted ,
         heart__bl_resids =   heart__observed - heart_bl__predicted,
         sleep__bl_resids =   sleep__observed - sleep_bl__predicted,
    
            
        # Compute excess (residuals) of model run on ALL days
         steps__resids =   steps__observed - steps__predicted,
         heart__resids =   heart__observed - heart__predicted,
         sleep__resids =   sleep__observed - sleep__predicted,
         
         steps__was_imputed = ifelse(valid_user__steps == 1 & (valid_day__steps != 1 | is.na(steps__observed)), 1, 0),
         heart__was_imputed = ifelse(valid_user__heart == 1 & (valid_day__heart != 1 | is.na(heart__observed)), 1, 0),
         sleep__was_imputed = ifelse(valid_user__sleep == 1 & (valid_day__sleep != 1 | is.na(sleep__observed)), 1, 0)
  ) %>%         

  # z-score imputed activity data columns by individual
  group_by(user_id) %>%
  mutate_at(vars(steps__imputed, sleep__imputed, heart__imputed, steps__observed, heart__observed, sleep__observed), list(z = scale2)) %>%
  ungroup()                      

# COMMAND ----------

# MAGIC %md ## Figure 5: Fraction of participants with elevated RHR
# MAGIC 
# MAGIC This code is in python so load saved activity data (containing imputed values) to reproduce the plot

# COMMAND ----------

# MAGIC %python 
# MAGIC #NOTE: UPDATE THIS CELL WITH THE APPROPRIATE PATHNAMES
# MAGIC 
# MAGIC #define pathname of dataset
# MAGIC activity_data_path = '/dbfs/data/covid19/manuscript_data_delivery_for_synapse/activity_data.csv.gz'

# COMMAND ----------

# MAGIC %python 
# MAGIC 
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC import scipy.stats as stats
# MAGIC 
# MAGIC import matplotlib.pyplot as plt
# MAGIC import seaborn as sns
# MAGIC import matplotlib.ticker as mtick
# MAGIC 
# MAGIC #NOTE: delete the plt.style line when running; plots will have different styling, but the same content
# MAGIC plt.style.use('/dbfs/FileStore/tables/eve.mplstyle')
# MAGIC [eve_yellow, eve_orange, eve_purple] = ['#f4c15d', '#ec8c46', '#823266']
# MAGIC 
# MAGIC activity_df = pd.read_csv(activity_data_path)
# MAGIC 
# MAGIC #subset to valid users for the heart stream 
# MAGIC activity_df = activity_df.loc[activity_df['valid_user__heart']==1]
# MAGIC 
# MAGIC #verify cohorts 
# MAGIC activity_df.drop_duplicates(subset=['user_id'])['cohort'].value_counts()

# COMMAND ----------

# MAGIC %python 
# MAGIC 
# MAGIC #define RHR thesholds at 0, 0.5, and 1 standard deviation above participants' mean RHR
# MAGIC thresholds = [0, 0.5, 1]
# MAGIC 
# MAGIC def compute_proportion_variance(x):
# MAGIC   #(p(1-p))/n
# MAGIC   p = x.mean()
# MAGIC   n = len(x)
# MAGIC   return (p * (1-p)) / n
# MAGIC 
# MAGIC fig, axes = plt.subplots(1, 3, figsize=(12,4), dpi=800, sharex=True, sharey=True)
# MAGIC 
# MAGIC for cohort, ax in zip(['COVID-19', 'Non-COVID-19 Flu', 'Pre-COVID-19 Flu'], axes):
# MAGIC   
# MAGIC   #pull out z-scored RHR values for the cohort
# MAGIC   z_subset = activity_df.loc[activity_df['cohort']==cohort, ['days_since_symptoms_onset', 'heart__imputed_z', 'user_id']]
# MAGIC   z_subset = z_subset.sort_values(['user_id','days_since_symptoms_onset'])
# MAGIC   
# MAGIC   cohort_n = z_subset['user_id'].nunique()
# MAGIC   print(cohort_n)
# MAGIC 
# MAGIC   for threshold, thresh_color in zip(thresholds, [eve_yellow, eve_orange, eve_purple]):
# MAGIC     
# MAGIC     #compute proportion of cohort with rhr above the threshold and subset to ILI days -10 to 20 
# MAGIC     #apply a 3-day backward-looking rolling mean to the proportions
# MAGIC     daily_proportions = (z_subset.groupby(by='days_since_symptoms_onset')['heart__imputed_z']
# MAGIC                                  .apply(lambda x: (x>threshold).mean())
# MAGIC                                  .rolling(3)
# MAGIC                                  .mean()
# MAGIC                                  .loc[-10:20])
# MAGIC     
# MAGIC     #compute standard deviation associated with the smoothed proportions
# MAGIC     #proportion variance: (p(1-p))/n
# MAGIC     proportion_variance = (daily_proportions * (1-daily_proportions)) / cohort_n
# MAGIC 
# MAGIC     #plot smoothed proportions and the associated proportion standard deviations
# MAGIC     ax.errorbar(x=daily_proportions.index, y=daily_proportions.values, yerr=np.sqrt(proportion_variance)/2, 
# MAGIC                 color=thresh_color, ecolor=thresh_color+'77', elinewidth=1, marker='o', markersize=3, linewidth=1,)
# MAGIC     
# MAGIC     #format the plot
# MAGIC     ax.yaxis.grid(False)
# MAGIC     ax.set_ylabel('')
# MAGIC     
# MAGIC     #add a line at day 0
# MAGIC     ax.axvline(x=0, ls='-', color='black', lw=0.5, zorder=0)
# MAGIC         
# MAGIC     #plot a horizontal line at the expected proportion for context
# MAGIC     ax.axhline(y=1-stats.norm.cdf(threshold), ls=':', lw=2, color=thresh_color, alpha=0.4)
# MAGIC     
# MAGIC     #highlight ili days -2 to 2 
# MAGIC     ax.axvspan(-2.5, 2.5, color=[0.7, 0.8, 0.8], alpha=0.1, zorder=0)
# MAGIC     
# MAGIC   #label the subplots
# MAGIC   ax.set_title(cohort)
# MAGIC   ax.set_xlabel('Days Since ILI-Onset')
# MAGIC   
# MAGIC axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# MAGIC axes[0].set_ylabel('% Cohort with Elevated RHR \n (3-day rolling average)')
# MAGIC plt.tight_layout()
# MAGIC fig.patch.set_facecolor('white')
# MAGIC display(ax)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Fit separate GAMs for each Cohort and each activity data channel

# COMMAND ----------

steps_data <- model_outputs %>% filter(valid_user__steps ==1 & valid_day__steps == 1 & days_since_symptoms_onset >= -10 & days_since_symptoms_onset <= 20) %>% droplevels() %>% mutate(user_id = as.factor(user_id))
heart_data <- model_outputs %>% filter(valid_user__heart ==1 & valid_day__heart == 1 & days_since_symptoms_onset >= -10 & days_since_symptoms_onset <= 20) %>% droplevels() %>% mutate(user_id = as.factor(user_id))
sleep_data <- model_outputs %>% filter(valid_user__sleep ==1 & valid_day__sleep == 1 & days_since_symptoms_onset >= -10 & days_since_symptoms_onset <= 20) %>% droplevels() %>% mutate(user_id = as.factor(user_id))

# COMMAND ----------

rm(all_preds_gam_bl_resids)

for (channel in c("steps", "sleep")) {
  for (group in c("COVID-19", "Pre-COVID-19 Flu", "Non-COVID-19 Flu")) {
     
     # Set up data
     channel_dat <- get(paste0(channel, "_data"))
     model_dat <- channel_dat %>% 
      filter(cohort==group) %>% 
      droplevels() %>% 
      mutate(y = !!as.name((paste0(channel, "__bl_resids"))))
     
     # Fit gam
     model_fit <-  mgcv::gam(y ~ s(days_since_symptoms_onset) + s(user_id, bs="re"), data=model_dat, method = 'REML')
     # Print gam results 
     print(channel); print(group); print(summary(model_fit));
     
     # Store model predictions
     preds_dat <- model_dat %>% group_by(days_since_symptoms_onset, cohort) %>% summarise(mean__y = mean(y, na.rm=T)) %>% ungroup() %>% mutate(user_id = 1, y_channel = channel)
     cur_preds <- as.data.frame(predict(model_fit, exclude="s(id)", se.fit=TRUE, newdata= preds_dat))
     preds_dat <- cbind(preds_dat, cur_preds)
     
      #  Add predictions to df of all model predictions results
      ifelse(exists("all_preds_gam_bl_resids", envir = .GlobalEnv),
           all_preds_gam_bl_resids <<- full_join(all_preds_gam_bl_resids, preds_dat),
           assign("all_preds_gam_bl_resids", preds_dat, envir=.GlobalEnv))
  }
}

# COMMAND ----------

# MAGIC %md ## Figure 6: Deviations from typical healthy behavior and physiology during ILI events

# COMMAND ----------

options(repr.plot.width = 2600, repr.plot.height = 1800)

# COMMAND ----------

all_preds_gam_bl_resids %>%
  mutate(cohort = fct_relevel(cohort, "COVID-19",  "Non-COVID-19 Flu",  "Pre-COVID-19 Flu"),
         y_channel = fct_rev(fct_recode(y_channel, 
                                 "Excess in Daily\nNumber of Steps" = "steps", 
                                 "Excess in Daily\nSleep (minutes)" = "sleep"))) %>%
  ggplot(aes(x = days_since_symptoms_onset, 
             y = fit,
             ymin = fit - se.fit, 
             ymax = fit + se.fit,
             group = cohort, 
             colour = cohort, 
             fill = cohort))+
    geom_vline(xintercept = 0) + 
    geom_hline(yintercept = 0) +
    geom_point(aes(y = mean__y), size = 1) +
    geom_ribbon(alpha = .3, colour = NA) +
    geom_line(size = .5) +
    facet_grid(y_channel ~ cohort, scales = "free", switch = "y") +
    labs(y = "Excess (observed - predicted)\n error band = 1 SE",  x = "Days since Illness Onset") +
    apply_theme_basic(cohort_palette) +
    guides(colour = F, fill = F)

# COMMAND ----------

# MAGIC %md ## Supplementary Figure: Sensor Data Coverage 

# COMMAND ----------

options(repr.plot.width = 2000, repr.plot.height = 2000)

# COMMAND ----------

user_validity <- analysis_panel %>% select(user_id, valid_user__heart, valid_user__steps, valid_user__sleep) %>% unique() %>% droplevels() %>%
  #filter(valid_user__heart==1 | valid_user__steps ==1 | valid_user__sleep == 1 ) %>% droplevels() %>%
  gather(channel, user_validity, -user_id) %>%
  separate(channel, into = c(NA, "channel"), sep="__") 

user_ranks <- analysis_panel %>%
  filter(valid_user__heart==1 | valid_user__steps ==1 | valid_user__sleep == 1 ) %>% droplevels() %>%
  filter(!is.na(cohort)) %>% droplevels() %>%
  dplyr::select(user_id, cohort, date_onset_merged) %>% unique() %>% 
  mutate(user_rank_valid = row_number(date_onset_merged)) %>%
  mutate(user_rank_valid = ifelse(cohort=="COVID-19", -(1000+user_rank_valid), ifelse(cohort=="Non-COVID-19 Flu", -(2000+user_rank_valid), -user_rank_valid))) %>%
  mutate(user_rank_valid = row_number(user_rank_valid)) %>%
  ungroup() %>% dplyr::select(user_id, user_rank_valid)

analysis_panel %>%
  filter(valid_user__heart==1 | valid_user__steps ==1 | valid_user__sleep == 1 ) %>% droplevels() %>%
  filter(!is.na(cohort)) %>% droplevels() %>%
  mutate(is_onset = ifelse(days_since_symptoms_onset==0, 1, 0)) %>%
  dplyr::select(user_id, cohort, study_date, is_onset, date_onset_merged, valid_day__steps, valid_day__heart, valid_day__sleep) %>%
  mutate(study_date = as.Date(study_date)) %>%
  gather(channel, day_validity, -user_id, -study_date, -is_onset, -cohort, -date_onset_merged) %>%
  separate(channel, into = c(NA, "channel"), sep="__") %>%
  full_join(user_ranks) %>%
  full_join(user_validity) %>%
  filter(!is.na(cohort) & user_validity==1) %>% droplevels() %>%
  mutate(day_type = ifelse(is_onset == 1, "Symptoms Onset",
                           ifelse(day_validity == 0, "no observation",
                                 ifelse(day_validity == 1,
                                       ifelse(cohort == "COVID-19", "COVID-19 observation", 
                                               ifelse(cohort == "Pre-COVID-19 Flu", "Pre-COVID-19 Flu observation", 
                                                      ifelse(cohort == "Non-COVID-19 Flu", "Non-COVID-19 Flu observation", NA ))),
                                       NA))),
        day_type = fct_relevel(day_type, "Symptoms Onset", "COVID-19 observation",  "Pre-COVID-19 Flu observation",  "Non-COVID-19 Flu observation", "no observation"),
        channel = fct_recode(channel, "RHR" = "heart", "Steps" = "steps", "Sleep" = "sleep"),
        channel = fct_relevel(channel, "Steps", "RHR", "Sleep")) %>%


  ggplot(aes(x = study_date, y = user_rank_valid, fill = as.factor(day_type)))+
    geom_raster()+
    facet_wrap(~channel) +
    scale_x_date(date_breaks = "1 month",
             date_labels = "%B") +
    labs(y = "Participant sorted by date of symptoms onset", x = "date") +
    guides(fill=guide_legend(title="valid day"))+
    theme_classic() + 
    scale_y_continuous(expand = c(0,0))+
    apply_theme_density(c("#F4C15D", "#287c85",  "#b8b8b8", "#787878",  "white"))

# COMMAND ----------


