% Statistical Analysis
clear all
close all
clc

main_dir_data = 'C:\Users\serena.moscato4\OneDrive - Alma Mater Studiorum Università di Bologna\Personal_Health_Systems_Lab\IgorDiemberger\Hom3ostasis\Acq\NewAnalysis2023\CinC\Data';
features_lab = readtable(fullfile(main_dir_data,'features_lab.csv'));
features_realworld_day = readtable(fullfile(main_dir_data,'features_realWorld_DAY.csv')); 
features_realworld_night = readtable(fullfile(main_dir_data,'features_realWorld_NIGHT.csv'));

hypertension = features_lab.Hypertension;
lowBP_ind = find(hypertension==0);
highBP_ind = find(hypertension==1);

features_lab.Tpi = [];

pvalue_condition = zeros(size(features_lab,2)-2,10);
pvalue_BP = zeros(size(features_lab,2)-2,3);

% Don't use the Tpi features, so start from 2
for i = 2:size(features_lab,2)-1

    figure
    tiledlayout(1,3);
    a1 = nexttile;
    boxchart(table2array(features_lab(:,i)),'GroupBy',hypertension);
    title('Lab')
    xticklabels('')
    a2 = nexttile;
    boxchart(table2array(features_realworld_day(:,i)),'GroupBy',hypertension);
    title('Real World - Day')
    xticklabels('')
    a3 = nexttile;
    boxchart(table2array(features_realworld_night(:,i)),'GroupBy',hypertension);
    title('Real World - Night')
    xticklabels('')
    legend('low BP','high BP')
    sgtitle(char(features_lab.Properties.VariableNames(i)))
    limits_a1 = a1.YAxis.Limits; 
    limits_a2 = a2.YAxis.Limits; 
    limits_a3 = a3.YAxis.Limits; 

    ylim_min = min([limits_a1(1), limits_a2(1), limits_a3(1)]);
    ylim_max = max([limits_a1(2), limits_a2(2), limits_a3(2)]);

    a1.YLim = [ylim_min ylim_max];
    a2.YLim = [ylim_min ylim_max];
    a3.YLim = [ylim_min ylim_max];

    % Repeated Measures ANOVA to compare lab, real world day and real world
    % night
    mat_var = [table2array(features_lab(:,i)), table2array(features_realworld_day(:,i)), table2array(features_realworld_night(:,i))];
    tab_var = array2table(mat_var); 
    tab_var.Properties.VariableNames = {'Lab','RWDay','RWNight'};

    withinDesign = table([1 2 3]','VariableNames',{'Condition'}); 
    withinDesign.Condition = categorical(withinDesign.Condition);
    
    rm = fitrm(tab_var,'Lab-RWNight ~ 1','WithinDesign',withinDesign);
    AT = ranova(rm,'WithinModel','Condition'); 

    pvalue_condition(i,1) = AT.pValue(3);

    % Paired ttest - couple comparison
    [~,pvalue_condition(i,2)] = ttest(table2array(tab_var(:,1)),table2array(tab_var(:,2)));
    [~,pvalue_condition(i,3)] = ttest(table2array(tab_var(:,1)),table2array(tab_var(:,3))); 
    [~,pvalue_condition(i,4)] = ttest(table2array(tab_var(:,2)),table2array(tab_var(:,3)));

    % Paired ttest - couple comparison dividing for BP
    [~,pvalue_condition(i,5)] = ttest(table2array(tab_var(lowBP_ind,1)),table2array(tab_var(lowBP_ind,2)));
    [~,pvalue_condition(i,6)] = ttest(table2array(tab_var(lowBP_ind,1)),table2array(tab_var(lowBP_ind,3))); 
    [~,pvalue_condition(i,7)] = ttest(table2array(tab_var(lowBP_ind,2)),table2array(tab_var(lowBP_ind,3)));

    [~,pvalue_condition(i,8)] = ttest(table2array(tab_var(highBP_ind,1)),table2array(tab_var(highBP_ind,2)));
    [~,pvalue_condition(i,9)] = ttest(table2array(tab_var(highBP_ind,1)),table2array(tab_var(highBP_ind,3))); 
    [~,pvalue_condition(i,10)] = ttest(table2array(tab_var(highBP_ind,2)),table2array(tab_var(highBP_ind,3)));

    % Comparison within the same condition between lowBP and highBP
    [~,pvalue_BP(i,1)] = ttest2(table2array(features_lab(lowBP_ind,i)),table2array(features_lab(highBP_ind,i)));
    [~,pvalue_BP(i,2)] = ttest2(table2array(features_realworld_day(lowBP_ind,i)),table2array(features_realworld_day(highBP_ind,i)));
    [~,pvalue_BP(i,3)] = ttest2(table2array(features_realworld_night(lowBP_ind,i)),table2array(features_realworld_night(highBP_ind,i)));
    
end
%%

pvalue_condition(1,:) = [];

pvalue_condition_tab = array2table(pvalue_condition);
pvalue_condition_tab.Properties.VariableNames = {'p-value ANOVA','p-value Lab-RWDay','pvalue Lab-RWNight','pvalue RWDay-RWNight','p-value LowBP Lab-RWDay','pvalue LowBP Lab-RWNight','pvalue LowBP RWDay-RWNight','p-value HighBP Lab-RWDay','pvalue HighBP Lab-RWNight','pvalue HighBP RWDay-RWNight'};
pvalue_condition_tab.Properties.RowNames = features_lab.Properties.VariableNames(2:end-1);

pvalue_BP(1,:) = [];

pvalue_BP_tab = array2table(pvalue_BP);
pvalue_BP_tab.Properties.VariableNames = {'Lab','RWDay','RWNight'};
pvalue_BP_tab.Properties.RowNames = features_lab.Properties.VariableNames(2:end-1);

writetable(pvalue_condition_tab,fullfile(main_dir_data,'Results_Conditions.xlsx'),"WriteRowNames",true);
writetable(pvalue_BP_tab,fullfile(main_dir_data,'Results_BP.xlsx'),"WriteRowNames",true);
