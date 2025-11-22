/*****************************************************************************
 * PROGRAM: nvdrs_train_suicide_analysis.sas
 * PURPOSE: Logistic regression analysis of train suicide risk factors
 * DATA SOURCE: NVDRS (National Violent Death Reporting System)
 * AUTHOR: Public Health Research
 * DATE: 2025-11-22
 *
 * RESEARCH QUESTIONS:
 * 1. What are demographic trends in train suicides (rates by age, gender, race)?
 * 2. What risk factors predict train vs. other suicide methods?
 * 3. How do socioeconomic factors contribute to train suicide risk?
 *****************************************************************************/

/* Set global options */
OPTIONS FORMCHAR="|----|+|---+=|-/\<>*" NODATE NONUMBER;
TITLE;
FOOTNOTE;

/* Define library paths - MODIFY THESE FOR YOUR ENVIRONMENT */
LIBNAME nvdrs "/path/to/nvdrs/data";  /* Raw NVDRS data */
LIBNAME analysis "/path/to/analysis/output";  /* Analysis outputs */

/* Set ODS output destination */
ODS HTML PATH="/path/to/output" FILE="train_suicide_analysis.html";
ODS GRAPHICS ON / IMAGENAME="train_suicide_" RESET=INDEX;


/*****************************************************************************
 * SECTION 1: DATA PREPARATION
 *****************************************************************************/

TITLE "NVDRS Train Suicide Analysis - Data Preparation";

/* Load and prepare NVDRS dataset */
DATA work.nvdrs_clean;
    SET nvdrs.restricted_use_data;  /* Use appropriate NVDRS file */

    /* OUTCOME VARIABLE: Train suicide indicator */
    /* NVDRS WeaponType1 codes or ICD-10 codes for train/railway */
    IF DeathManner = 'Suicide' THEN DO;
        /* Create binary outcome: 1=train suicide, 0=other suicide method */
        IF WeaponType1 IN ('Train', 'Railway') OR
           ICD10Code IN ('X81.0', 'X81.1', 'X81.8', 'X81.9') OR
           INDEX(UPCASE(CMENotes), 'TRAIN') > 0 OR
           INDEX(UPCASE(CMENotes), 'RAILWAY') > 0 OR
           INDEX(UPCASE(CMENotes), 'RAILROAD') > 0 THEN train_suicide = 1;
        ELSE train_suicide = 0;
    END;
    ELSE DELETE;  /* Keep only suicides */

    /* DEMOGRAPHIC VARIABLES */

    /* Age groups */
    IF Age < 18 THEN age_group = '1_Under18';
    ELSE IF Age >= 18 AND Age < 25 THEN age_group = '2_18-24';
    ELSE IF Age >= 25 AND Age < 35 THEN age_group = '3_25-34';
    ELSE IF Age >= 35 AND Age < 45 THEN age_group = '4_35-44';
    ELSE IF Age >= 45 AND Age < 55 THEN age_group = '5_45-54';
    ELSE IF Age >= 55 AND Age < 65 THEN age_group = '6_55-64';
    ELSE IF Age >= 65 THEN age_group = '7_65plus';
    ELSE age_group = '9_Unknown';

    /* Gender (recode if needed) */
    IF Sex = 'M' THEN gender = 'Male';
    ELSE IF Sex = 'F' THEN gender = 'Female';
    ELSE gender = 'Unknown';

    /* Race/Ethnicity combined variable */
    IF Hispanic = 'Yes' THEN race_eth = '1_Hispanic';
    ELSE IF Race = 'White' THEN race_eth = '2_NH_White';
    ELSE IF Race = 'Black' THEN race_eth = '3_NH_Black';
    ELSE IF Race IN ('Asian', 'Pacific Islander') THEN race_eth = '4_NH_Asian_PI';
    ELSE IF Race = 'American Indian/Alaska Native' THEN race_eth = '5_NH_AIAN';
    ELSE race_eth = '6_Other_Unknown';

    /* Education level (SES proxy) */
    IF EducationLevel IN ('Less than HS', 'Some HS') THEN education = '1_Less_HS';
    ELSE IF EducationLevel = 'High School/GED' THEN education = '2_HS_GED';
    ELSE IF EducationLevel IN ('Some College', 'Associate') THEN education = '3_Some_College';
    ELSE IF EducationLevel IN ('Bachelor', 'Graduate') THEN education = '4_College_Plus';
    ELSE education = '9_Unknown';

    /* Marital status */
    IF MaritalStatus IN ('Married', 'Civil Union') THEN marital = '1_Married';
    ELSE IF MaritalStatus IN ('Never Married', 'Single') THEN marital = '2_Never_Married';
    ELSE IF MaritalStatus IN ('Divorced', 'Separated') THEN marital = '3_Divorced_Sep';
    ELSE IF MaritalStatus = 'Widowed' THEN marital = '4_Widowed';
    ELSE marital = '9_Unknown';

    /* RISK FACTORS FROM NVDRS CIRCUMSTANCE VARIABLES */

    /* Mental health history */
    mh_depression = (DepressedMood = 'Yes');
    mh_current_treatment = (CurrentMentalHealthTreatment = 'Yes');
    mh_history_treatment = (HistoryMentalHealthTreatment = 'Yes');
    mh_problem = (CurrentMentalHealthProblem = 'Yes');

    /* Substance use */
    substance_alcohol = (AlcoholProblem = 'Yes');
    substance_drug = (DrugAbuse = 'Yes');
    substance_positive = (ToxicologyPositive = 'Yes');

    /* Crisis factors */
    crisis_relationship = (IntimatePartnerProblem = 'Yes');
    crisis_job = (JobProblem = 'Yes');
    crisis_financial = (FinancialProblem = 'Yes');
    crisis_legal = (LegalProblem = 'Yes');
    crisis_physical_health = (PhysicalHealthProblem = 'Yes');

    /* Suicide history */
    history_attempt = (HistorySuicideAttempt = 'Yes');
    history_ideation = (SuicidalIdeation = 'Yes');
    disclosed_intent = (DisclosedIntent = 'Yes');
    left_note = (SuicideNote = 'Yes');

    /* Homelessness (SES indicator) */
    homeless = (Homeless = 'Yes');

    /* Veteran status */
    veteran = (Veteran = 'Yes');

    /* Geographic region */
    IF State IN ('CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA')
        THEN region = '1_Northeast';
    ELSE IF State IN ('IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD')
        THEN region = '2_Midwest';
    ELSE IF State IN ('DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX')
        THEN region = '3_South';
    ELSE IF State IN ('AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA')
        THEN region = '4_West';
    ELSE region = '9_Unknown';

    /* Year of death (for trend analysis) */
    year = YEAR(DeathDate);

    /* Keep only complete cases for main analysis */
    IF CMISS(OF Age gender race_eth education) = 0;

    /* Create analysis weight (if using weighted data) */
    IF weight = . THEN weight = 1;

    LABEL
        train_suicide = "Train/Railway Suicide (1=Yes, 0=Other Method)"
        age_group = "Age Group"
        gender = "Gender"
        race_eth = "Race/Ethnicity"
        education = "Education Level"
        marital = "Marital Status"
        region = "Geographic Region"
        year = "Year of Death"
        mh_depression = "Depressed Mood"
        mh_current_treatment = "Current Mental Health Treatment"
        mh_history_treatment = "History of Mental Health Treatment"
        substance_alcohol = "Alcohol Problem"
        substance_drug = "Drug Abuse Problem"
        crisis_relationship = "Relationship Problem"
        crisis_job = "Job Problem"
        crisis_financial = "Financial Problem"
        homeless = "Homeless"
        veteran = "Veteran Status"
        history_attempt = "Prior Suicide Attempt"
        left_note = "Left Suicide Note"
    ;
RUN;

/* Check data quality */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES train_suicide * (age_group gender race_eth education marital region) / MISSING;
    TITLE2 "Data Quality Check - Missing Values";
RUN;

PROC MEANS DATA=work.nvdrs_clean N NMISS MEAN STD MIN MAX;
    VAR Age year train_suicide mh_depression substance_alcohol crisis_financial;
    TITLE2 "Descriptive Statistics for Key Variables";
RUN;


/*****************************************************************************
 * SECTION 2: DESCRIPTIVE STATISTICS & TRENDS
 *****************************************************************************/

TITLE "SECTION 2: Descriptive Analysis of Train Suicides";

/* Overall counts and rates */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES train_suicide / NOCUM;
    TITLE2 "Overall Train Suicide Prevalence";
RUN;

/* Temporal trends */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES year * train_suicide / NOCOL NOPERCENT;
    TITLE2 "Train Suicide Counts by Year";
RUN;

PROC SGPLOT DATA=work.nvdrs_clean;
    VBAR year / RESPONSE=train_suicide STAT=mean;
    YAXIS LABEL="Proportion of Train Suicides";
    XAXIS LABEL="Year";
    TITLE2 "Temporal Trend in Train Suicide Proportion";
RUN;

/* Demographics: Age */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES age_group * train_suicide / CHISQ RELRISK;
    TITLE2 "Train Suicide by Age Group";
RUN;

/* Demographics: Gender */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES gender * train_suicide / CHISQ RELRISK;
    TITLE2 "Train Suicide by Gender";
RUN;

/* Demographics: Race/Ethnicity */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES race_eth * train_suicide / CHISQ RELRISK;
    TITLE2 "Train Suicide by Race/Ethnicity";
RUN;

/* Socioeconomic: Education */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES education * train_suicide / CHISQ RELRISK;
    WHERE education NE '9_Unknown';
    TITLE2 "Train Suicide by Education Level (SES Proxy)";
RUN;

/* Socioeconomic: Marital Status */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES marital * train_suicide / CHISQ RELRISK;
    WHERE marital NE '9_Unknown';
    TITLE2 "Train Suicide by Marital Status";
RUN;

/* Geographic: Region */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES region * train_suicide / CHISQ RELRISK;
    TITLE2 "Train Suicide by Geographic Region";
RUN;

/* Risk Factors: Mental Health */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES (mh_depression mh_current_treatment mh_history_treatment) * train_suicide
           / CHISQ RELRISK NOCOL NOPERCENT;
    TITLE2 "Train Suicide by Mental Health Indicators";
RUN;

/* Risk Factors: Substance Use */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES (substance_alcohol substance_drug) * train_suicide
           / CHISQ RELRISK NOCOL NOPERCENT;
    TITLE2 "Train Suicide by Substance Use";
RUN;

/* Risk Factors: Crisis Circumstances */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES (crisis_relationship crisis_job crisis_financial crisis_physical_health) * train_suicide
           / CHISQ RELRISK NOCOL NOPERCENT;
    TITLE2 "Train Suicide by Crisis Circumstances";
RUN;

/* Risk Factors: Homelessness */
PROC FREQ DATA=work.nvdrs_clean;
    TABLES homeless * train_suicide / CHISQ RELRISK;
    TITLE2 "Train Suicide by Homeless Status (SES Indicator)";
RUN;


/*****************************************************************************
 * SECTION 3: UNIVARIATE LOGISTIC REGRESSION MODELS
 *****************************************************************************/

TITLE "SECTION 3: Univariate Logistic Regression Models";

/* Create macro for univariate analysis */
%MACRO univariate_logistic(var, var_label);
    PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
        CLASS &var / PARAM=REF;
        MODEL train_suicide(EVENT='1') = &var;
        ODS OUTPUT ParameterEstimates=work.param_&var
                   OddsRatios=work.or_&var;
        TITLE2 "Univariate Model: &var_label";
    RUN;
%MEND;

/* Run univariate models for all predictors */
%univariate_logistic(age_group, Age Group);
%univariate_logistic(gender, Gender);
%univariate_logistic(race_eth, Race/Ethnicity);
%univariate_logistic(education, Education Level);
%univariate_logistic(marital, Marital Status);
%univariate_logistic(region, Geographic Region);
%univariate_logistic(mh_depression, Depressed Mood);
%univariate_logistic(mh_current_treatment, Current MH Treatment);
%univariate_logistic(substance_alcohol, Alcohol Problem);
%univariate_logistic(substance_drug, Drug Problem);
%univariate_logistic(crisis_relationship, Relationship Problem);
%univariate_logistic(crisis_job, Job Problem);
%univariate_logistic(crisis_financial, Financial Problem);
%univariate_logistic(homeless, Homeless);
%univariate_logistic(veteran, Veteran);
%univariate_logistic(history_attempt, Prior Suicide Attempt);
%univariate_logistic(left_note, Left Suicide Note);

/* Compile univariate results */
DATA work.univariate_summary;
    LENGTH variable $50 level $50;
    SET work.or_age_group (IN=a)
        work.or_gender (IN=b)
        work.or_race_eth (IN=c)
        work.or_education (IN=d)
        work.or_marital (IN=e)
        work.or_region (IN=f)
        work.or_mh_depression (IN=g)
        work.or_substance_alcohol (IN=h)
        work.or_crisis_financial (IN=i)
        work.or_homeless (IN=j)
        work.or_veteran (IN=k)
        work.or_history_attempt (IN=l);

    IF a THEN variable = "Age Group";
    ELSE IF b THEN variable = "Gender";
    ELSE IF c THEN variable = "Race/Ethnicity";
    ELSE IF d THEN variable = "Education";
    ELSE IF e THEN variable = "Marital Status";
    ELSE IF f THEN variable = "Region";
    ELSE IF g THEN variable = "Depressed Mood";
    ELSE IF h THEN variable = "Alcohol Problem";
    ELSE IF i THEN variable = "Financial Problem";
    ELSE IF j THEN variable = "Homeless";
    ELSE IF k THEN variable = "Veteran";
    ELSE IF l THEN variable = "Prior Attempt";

    level = Effect;
    OR = OddsRatioEst;
    OR_LCL = LowerCL;
    OR_UCL = UpperCL;

    KEEP variable level OR OR_LCL OR_UCL;
RUN;

PROC PRINT DATA=work.univariate_summary NOOBS;
    TITLE2 "Summary of Univariate Odds Ratios";
    FORMAT OR OR_LCL OR_UCL 5.3;
RUN;


/*****************************************************************************
 * SECTION 4: MULTIVARIABLE LOGISTIC REGRESSION MODELS
 *****************************************************************************/

TITLE "SECTION 4: Multivariable Logistic Regression Models";

/* MODEL 1: Demographics Only */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group (REF='3_25-34')
          gender (REF='Male')
          race_eth (REF='2_NH_White')
          region (REF='3_South')
          / PARAM=REF;
    MODEL train_suicide(EVENT='1') = age_group gender race_eth region /
          CLODDS=WALD LACKFIT;
    ODS OUTPUT ParameterEstimates=work.model1_params
               OddsRatios=work.model1_or
               Association=work.model1_fit;
    TITLE2 "Model 1: Demographics (Age, Gender, Race, Region)";
RUN;

/* MODEL 2: Demographics + Socioeconomic Status */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group (REF='3_25-34')
          gender (REF='Male')
          race_eth (REF='2_NH_White')
          education (REF='2_HS_GED')
          marital (REF='1_Married')
          homeless (REF='0')
          region (REF='3_South')
          / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education marital homeless region /
          CLODDS=WALD LACKFIT;
    ODS OUTPUT ParameterEstimates=work.model2_params
               OddsRatios=work.model2_or
               Association=work.model2_fit;
    TITLE2 "Model 2: Demographics + SES (Education, Marital Status, Homeless)";
RUN;

/* MODEL 3: Demographics + SES + Mental Health */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group (REF='3_25-34')
          gender (REF='Male')
          race_eth (REF='2_NH_White')
          education (REF='2_HS_GED')
          marital (REF='1_Married')
          homeless (REF='0')
          mh_depression (REF='0')
          mh_current_treatment (REF='0')
          region (REF='3_South')
          / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education marital homeless
                                      mh_depression mh_current_treatment region /
          CLODDS=WALD LACKFIT;
    ODS OUTPUT ParameterEstimates=work.model3_params
               OddsRatios=work.model3_or
               Association=work.model3_fit;
    TITLE2 "Model 3: Demographics + SES + Mental Health";
RUN;

/* MODEL 4: Full Model with Risk Factors */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group (REF='3_25-34')
          gender (REF='Male')
          race_eth (REF='2_NH_White')
          education (REF='2_HS_GED')
          marital (REF='1_Married')
          homeless (REF='0')
          mh_depression (REF='0')
          mh_current_treatment (REF='0')
          substance_alcohol (REF='0')
          substance_drug (REF='0')
          crisis_relationship (REF='0')
          crisis_job (REF='0')
          crisis_financial (REF='0')
          history_attempt (REF='0')
          veteran (REF='0')
          region (REF='3_South')
          / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education marital homeless
                                      mh_depression mh_current_treatment
                                      substance_alcohol substance_drug
                                      crisis_relationship crisis_job crisis_financial
                                      history_attempt veteran region /
          CLODDS=WALD LACKFIT SELECTION=NONE;
    ODS OUTPUT ParameterEstimates=work.model4_params
               OddsRatios=work.model4_or
               Association=work.model4_fit;
    TITLE2 "Model 4: Full Model with All Risk Factors";
RUN;

/* MODEL 5: Stepwise Selection Model */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group (REF='3_25-34')
          gender (REF='Male')
          race_eth (REF='2_NH_White')
          education (REF='2_HS_GED')
          marital (REF='1_Married')
          homeless (REF='0')
          mh_depression (REF='0')
          mh_current_treatment (REF='0')
          substance_alcohol (REF='0')
          substance_drug (REF='0')
          crisis_relationship (REF='0')
          crisis_job (REF='0')
          crisis_financial (REF='0')
          history_attempt (REF='0')
          veteran (REF='0')
          region (REF='3_South')
          / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education marital homeless
                                      mh_depression mh_current_treatment
                                      substance_alcohol substance_drug
                                      crisis_relationship crisis_job crisis_financial
                                      history_attempt veteran region /
          CLODDS=WALD SELECTION=STEPWISE SLENTRY=0.10 SLSTAY=0.05;
    ODS OUTPUT ParameterEstimates=work.model5_params
               OddsRatios=work.model5_or
               Association=work.model5_fit;
    TITLE2 "Model 5: Stepwise Selection Model (Entry=0.10, Stay=0.05)";
RUN;


/*****************************************************************************
 * SECTION 5: MODEL COMPARISON & DIAGNOSTICS
 *****************************************************************************/

TITLE "SECTION 5: Model Comparison and Diagnostics";

/* Compare model fit statistics */
DATA work.model_comparison;
    LENGTH model $50;
    SET work.model1_fit (IN=a)
        work.model2_fit (IN=b)
        work.model3_fit (IN=c)
        work.model4_fit (IN=d)
        work.model5_fit (IN=e);

    IF a THEN model = "1_Demographics";
    ELSE IF b THEN model = "2_Demographics+SES";
    ELSE IF c THEN model = "3_Demo+SES+MH";
    ELSE IF d THEN model = "4_Full_Model";
    ELSE IF e THEN model = "5_Stepwise";

    KEEP model nValue2;
    RENAME nValue2 = c_statistic;
RUN;

PROC PRINT DATA=work.model_comparison NOOBS;
    TITLE2 "C-Statistics (AUC) for Model Comparison";
    FORMAT c_statistic 5.4;
RUN;

/* ROC curve for best model */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING PLOTS(ONLY)=ROC;
    CLASS age_group gender race_eth education marital homeless
          mh_depression substance_alcohol crisis_financial history_attempt
          / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education marital homeless
                                      mh_depression substance_alcohol crisis_financial
                                      history_attempt / OUTROC=work.roc_data;
    TITLE2 "ROC Curve for Selected Model";
RUN;

/* Hosmer-Lemeshow Goodness-of-Fit Test */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group gender race_eth education marital homeless
          mh_depression substance_alcohol crisis_financial
          / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education marital homeless
                                      mh_depression substance_alcohol crisis_financial /
          LACKFIT;
    TITLE2 "Hosmer-Lemeshow Goodness-of-Fit Test";
RUN;

/* Check for multicollinearity */
PROC REG DATA=work.nvdrs_clean;
    MODEL train_suicide = Age / VIF TOL;
    TITLE2 "Variance Inflation Factors (Continuous Predictors)";
RUN;
QUIT;


/*****************************************************************************
 * SECTION 6: STRATIFIED ANALYSES
 *****************************************************************************/

TITLE "SECTION 6: Stratified Analyses by Demographics";

/* Stratified by Gender */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    BY gender;
    CLASS age_group race_eth education marital homeless mh_depression / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown' AND gender NE 'Unknown';
    MODEL train_suicide(EVENT='1') = age_group race_eth education marital homeless mh_depression;
    ODS OUTPUT OddsRatios=work.or_by_gender;
    TITLE2 "Stratified Model by Gender";
RUN;

/* Stratified by Age Group */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    BY age_group;
    CLASS gender race_eth education marital homeless mh_depression / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown' AND age_group NOT IN ('9_Unknown');
    MODEL train_suicide(EVENT='1') = gender race_eth education marital homeless mh_depression;
    ODS OUTPUT OddsRatios=work.or_by_age;
    TITLE2 "Stratified Model by Age Group";
RUN;

/* Stratified by Race/Ethnicity */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    BY race_eth;
    CLASS age_group gender education marital homeless mh_depression / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown' AND race_eth NE '6_Other_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender education marital homeless mh_depression;
    ODS OUTPUT OddsRatios=work.or_by_race;
    TITLE2 "Stratified Model by Race/Ethnicity";
RUN;


/*****************************************************************************
 * SECTION 7: INTERACTION ANALYSES
 *****************************************************************************/

TITLE "SECTION 7: Interaction Effect Analyses";

/* Test Age * Gender interaction */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group (REF='3_25-34')
          gender (REF='Male')
          race_eth (REF='2_NH_White')
          education (REF='2_HS_GED')
          / PARAM=REF;
    WHERE education NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group|gender race_eth education /
          CLODDS=WALD;
    TITLE2 "Interaction: Age Group * Gender";
RUN;

/* Test Gender * Mental Health interaction */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group (REF='3_25-34')
          gender (REF='Male')
          mh_depression (REF='0')
          / PARAM=REF;
    MODEL train_suicide(EVENT='1') = age_group gender|mh_depression /
          CLODDS=WALD;
    TITLE2 "Interaction: Gender * Depression";
RUN;

/* Test Homeless * Financial Crisis interaction */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group (REF='3_25-34')
          gender (REF='Male')
          homeless (REF='0')
          crisis_financial (REF='0')
          / PARAM=REF;
    MODEL train_suicide(EVENT='1') = age_group gender homeless|crisis_financial /
          CLODDS=WALD;
    TITLE2 "Interaction: Homeless * Financial Crisis";
RUN;


/*****************************************************************************
 * SECTION 8: PREDICTED PROBABILITIES & RISK SCORING
 *****************************************************************************/

TITLE "SECTION 8: Predicted Probabilities and Risk Scoring";

/* Generate predicted probabilities from final model */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group (REF='3_25-34')
          gender (REF='Male')
          race_eth (REF='2_NH_White')
          education (REF='2_HS_GED')
          marital (REF='1_Married')
          homeless (REF='0')
          mh_depression (REF='0')
          substance_alcohol (REF='0')
          crisis_financial (REF='0')
          / PARAM=REF;
    WHERE education NE '9_Unknown' AND marital NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education marital homeless
                                      mh_depression substance_alcohol crisis_financial;
    OUTPUT OUT=work.predictions PREDICTED=pred_prob LOWER=lcl_prob UPPER=ucl_prob;
    TITLE2 "Final Model with Predicted Probabilities";
RUN;

/* Analyze distribution of predicted probabilities */
PROC MEANS DATA=work.predictions N MEAN STD MIN P25 MEDIAN P75 MAX;
    CLASS train_suicide;
    VAR pred_prob;
    TITLE2 "Distribution of Predicted Probabilities by Actual Outcome";
RUN;

/* Create risk categories */
DATA work.risk_categories;
    SET work.predictions;

    IF pred_prob < 0.01 THEN risk_category = '1_Very_Low';
    ELSE IF pred_prob < 0.02 THEN risk_category = '2_Low';
    ELSE IF pred_prob < 0.05 THEN risk_category = '3_Moderate';
    ELSE IF pred_prob < 0.10 THEN risk_category = '4_High';
    ELSE risk_category = '5_Very_High';
RUN;

PROC FREQ DATA=work.risk_categories;
    TABLES risk_category * train_suicide / NOROW NOCOL;
    TITLE2 "Classification Table by Risk Category";
RUN;


/*****************************************************************************
 * SECTION 9: SENSITIVITY ANALYSES
 *****************************************************************************/

TITLE "SECTION 9: Sensitivity Analyses";

/* Sensitivity: Exclude missing education/marital */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group gender race_eth education marital / PARAM=REF;
    WHERE education NOT IN ('9_Unknown', '') AND marital NOT IN ('9_Unknown', '');
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education marital;
    TITLE2 "Sensitivity: Complete Case Analysis (No Missing SES)";
RUN;

/* Sensitivity: Time period restriction (recent years) */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group gender race_eth education / PARAM=REF;
    WHERE year >= 2018 AND education NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education;
    TITLE2 "Sensitivity: Recent Years Only (2018+)";
RUN;

/* Sensitivity: Age-restricted analysis (exclude elderly) */
PROC LOGISTIC DATA=work.nvdrs_clean DESCENDING;
    CLASS age_group gender race_eth education / PARAM=REF;
    WHERE age_group NOT IN ('7_65plus', '9_Unknown') AND education NE '9_Unknown';
    MODEL train_suicide(EVENT='1') = age_group gender race_eth education;
    TITLE2 "Sensitivity: Under 65 Years Only";
RUN;


/*****************************************************************************
 * SECTION 10: EXPORT RESULTS
 *****************************************************************************/

TITLE "SECTION 10: Export Results for Publication";

/* Export odds ratios from final model */
PROC EXPORT DATA=work.model4_or
    OUTFILE="/path/to/output/train_suicide_odds_ratios.csv"
    DBMS=CSV REPLACE;
RUN;

/* Export predicted probabilities */
PROC EXPORT DATA=work.predictions
    OUTFILE="/path/to/output/train_suicide_predictions.csv"
    DBMS=CSV REPLACE;
RUN;

/* Create summary table for publication */
DATA work.publication_table;
    SET work.model4_or;

    /* Format for publication */
    OR_CI = CATS(PUT(OddsRatioEst, 5.2), " (",
                 PUT(LowerCL, 5.2), "-",
                 PUT(UpperCL, 5.2), ")");

    IF ProbChiSq < 0.001 THEN p_value = "<0.001";
    ELSE IF ProbChiSq < 0.01 THEN p_value = "<0.01";
    ELSE IF ProbChiSq < 0.05 THEN p_value = "<0.05";
    ELSE p_value = PUT(ProbChiSq, 5.3);

    KEEP Effect OR_CI p_value;
RUN;

PROC PRINT DATA=work.publication_table NOOBS;
    TITLE2 "Formatted Results for Publication (Model 4)";
RUN;


/* Close ODS destinations */
ODS HTML CLOSE;
ODS GRAPHICS OFF;

TITLE;
FOOTNOTE;

/*****************************************************************************
 * END OF PROGRAM
 *
 * OUTPUTS GENERATED:
 * 1. HTML report: train_suicide_analysis.html
 * 2. CSV exports: train_suicide_odds_ratios.csv, train_suicide_predictions.csv
 * 3. SAS datasets: work.model1-5_or, work.predictions, work.risk_categories
 *
 * NEXT STEPS:
 * 1. Review model diagnostics (C-statistics, Hosmer-Lemeshow)
 * 2. Examine stratified analyses for effect modification
 * 3. Validate using cross-validation or holdout sample
 * 4. Consider survival analysis for time-to-event outcomes
 *****************************************************************************/
