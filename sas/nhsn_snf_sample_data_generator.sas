/*****************************************************************************
* Program: nhsn_snf_sample_data_generator.sas
* Purpose: Generate Sample NHSN SNF Data for QA Testing
* Author: Generated for Gen Z Agent Project
* Date: 2025-11-23
*
* Description:
*   Creates realistic sample NHSN SNF data with:
*   - Normal baseline values
*   - Injected anomalies and spikes
*   - Geographic distribution across HHS regions
*   - Multiple metric types
*****************************************************************************/

/* Configuration */
%let num_facilities = 500;
%let num_weeks = 16;
%let anomaly_rate = 0.05;  /* 5% of records will be anomalies */

/* Seed for reproducibility */
%let seed = 12345;

/* Step 1: Generate facility master data */
data work.facility_master;
    length facility_id $10 facility_name $100 state $2;

    array states[51] $2 _temporary_ (
        'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
        'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
        'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
        'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
        'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC'
    );

    /* HHS Region mapping (simplified) */
    array region_map[51] _temporary_ (
        4,10,9,6,9,8,1,3,4,4,
        9,10,5,5,7,7,4,6,1,3,
        1,5,5,4,7,8,7,9,1,2,
        6,2,4,8,5,6,10,3,1,4,
        8,4,6,8,1,3,10,3,5,8,3
    );

    call streaminit(&seed);

    do i = 1 to &num_facilities;
        facility_id = cats('SNF', put(i, z5.));

        /* Randomly assign state */
        state_idx = ceil(rand('uniform') * 51);
        state = states[state_idx];
        hhs_region = region_map[state_idx];

        /* Generate facility name */
        facility_name = catx(' ', state, 'Skilled Nursing Facility', put(i, 5.));

        /* Facility characteristics */
        total_beds = 50 + ceil(rand('uniform') * 200);  /* 50-250 beds */
        avg_occupancy_rate = 0.70 + (rand('uniform') * 0.25);  /* 70-95% */

        output;
    end;

    drop i state_idx;
run;

/* Step 2: Generate time series data */
data work.nhsn_snf_data;
    set work.facility_master;

    array metrics[5] $30 _temporary_ (
        'COVID_CASES',
        'COVID_DEATHS',
        'INFLUENZA_CASES',
        'BEDS_OCCUPIED',
        'STAFF_SHORTAGE'
    );

    /* Baseline parameters by metric type */
    array base_mean[5] _temporary_ (5, 0.3, 2, 0, 1);
    array base_std[5] _temporary_ (3, 0.5, 2, 0, 2);

    call streaminit(&seed + _n_);

    /* Generate weekly data for each facility */
    do week_num = 1 to &num_weeks;
        report_date = intnx('week', today(), -(&num_weeks - week_num), 'beginning');

        /* Generate data for each metric type */
        do metric_idx = 1 to 5;
            metric_type = metrics[metric_idx];

            /* Calculate population at risk */
            if metric_type = 'BEDS_OCCUPIED' then
                population_at_risk = total_beds;
            else
                population_at_risk = ceil(total_beds * avg_occupancy_rate);

            /* Generate baseline value with some random variation */
            if metric_type = 'BEDS_OCCUPIED' then do;
                /* Beds occupied is percentage */
                base_value = avg_occupancy_rate + (rand('normal') * 0.05);
                metric_value = ceil(total_beds * base_value);
            end;
            else do;
                /* Count data with seasonal trend */
                seasonal_factor = 1 + (0.3 * sin((week_num / 52) * 2 * constant('pi')));
                base_value = base_mean[metric_idx] * seasonal_factor;
                metric_value = max(0, ceil(base_value + (rand('normal') * base_std[metric_idx])));
            end;

            /* Inject anomalies for testing */
            inject_anomaly = (rand('uniform') < &anomaly_rate);

            if inject_anomaly then do;
                anomaly_type = ceil(rand('uniform') * 4);

                select(anomaly_type);
                    when(1) do;  /* High spike */
                        metric_value = metric_value * (3 + rand('uniform') * 5);
                        anomaly_label = 'HIGH_SPIKE';
                    end;
                    when(2) do;  /* Extreme spike */
                        metric_value = metric_value * (8 + rand('uniform') * 10);
                        anomaly_label = 'EXTREME_SPIKE';
                    end;
                    when(3) do;  /* Sudden drop */
                        metric_value = max(0, metric_value * 0.1);
                        anomaly_label = 'SUDDEN_DROP';
                    end;
                    when(4) do;  /* Impossible value */
                        metric_value = population_at_risk * (1.5 + rand('uniform'));
                        anomaly_label = 'EXCEEDS_POPULATION';
                    end;
                    otherwise;
                end;
            end;
            else do;
                anomaly_label = 'NORMAL';
            end;

            /* Ensure non-negative */
            metric_value = max(0, round(metric_value, 1));

            /* Data quality flags */
            if missing(metric_value) then data_quality = 'MISSING';
            else if metric_value < 0 then data_quality = 'NEGATIVE';
            else if metric_value > population_at_risk and metric_type ne 'BEDS_OCCUPIED'
                then data_quality = 'EXCEEDS_DENOMINATOR';
            else data_quality = 'VALID';

            output;
        end;
    end;

    format report_date yymmdd10.;
    keep facility_id facility_name state hhs_region report_date
         metric_type metric_value population_at_risk
         anomaly_label data_quality total_beds;
run;

/* Step 3: Add some missing data to test completeness checks */
data work.nhsn_snf_data;
    set work.nhsn_snf_data;

    call streaminit(&seed + 999);

    /* Randomly set some values to missing */
    if rand('uniform') < 0.02 then metric_value = .;
    if rand('uniform') < 0.01 then state = '';
    if rand('uniform') < 0.01 then hhs_region = .;
run;

/* Step 4: Data summary */
proc freq data=work.nhsn_snf_data;
    tables metric_type * anomaly_label / nocum;
    title "Sample Data Distribution - Anomaly Labels by Metric Type";
run;

proc means data=work.nhsn_snf_data n nmiss mean std min max;
    class metric_type;
    var metric_value population_at_risk;
    title "Sample Data Summary Statistics";
run;

proc sql;
    select
        count(*) as total_records,
        count(distinct facility_id) as total_facilities,
        count(distinct state) as total_states,
        count(distinct hhs_region) as total_regions,
        min(report_date) format=yymmdd10. as min_date,
        max(report_date) format=yymmdd10. as max_date
    from work.nhsn_snf_data;
quit;

title "Sample NHSN SNF Data Generated Successfully";
footnote "Total Facilities: &num_facilities | Weeks: &num_weeks | Anomaly Rate: %sysevalf(&anomaly_rate * 100)%";

/* Export sample data */
proc export data=work.nhsn_snf_data
    outfile="/home/user/GenZ/sas/sample_nhsn_snf_data.csv"
    dbms=csv replace;
run;

%put NOTE: Sample NHSN SNF data generated with &num_facilities facilities over &num_weeks weeks;
%put NOTE: Approximately %sysevalf(&anomaly_rate * 100)%% of records contain injected anomalies for testing;
