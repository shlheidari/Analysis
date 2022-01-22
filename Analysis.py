import pandas as pd
import numpy as np
import scipy
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from pingouin import pairwise_tukey

df_dependent = pd.read_excel('dependent.xlsx', header=0,sheet_name=0)
df_independent = pd.read_excel('independent.xlsx', header=0,sheet_name=0)

df_dependent=df_dependent.loc[:,~df_dependent.columns.str.match("Unnamed")]
df_independent=df_independent.loc[:,~df_independent.columns.str.match("Unnamed")]

df_dependent_int = df_dependent.select_dtypes(include='number')
df_dependent_str = df_dependent.select_dtypes(include='object')
df_independent_int = df_independent.select_dtypes(include='number')
df_independent_str = df_independent.select_dtypes(include='object')

ID_2=['count','unique','top','freq']
df_total_d_d=df_dependent_str.describe(include='all')
df_total_d_d.insert(0, "column_name", ID_2, True)
df_total_d_i=df_independent_str.describe(include='all')
df_total_d_i.insert(0, "column_name", ID_2, True)
df_total_d_ss = pd.merge(df_total_d_i, df_total_d_d ,how="outer", on="column_name")
ID_2=['count','mean','std','min','25%','50%','75%','max']
df_total_d_d=df_dependent_int.describe(include='all')
df_total_d_d.insert(0, "column_name", ID_2, True)
df_total_d_i=df_independent_int.describe(include='all')
df_total_d_i.insert(0, "column_name", ID_2, True)
df_total_d_ii = pd.merge(df_total_d_i, df_total_d_d ,how="outer", on="column_name")
df_total_d = pd.merge(df_total_d_ii, df_total_d_ss ,how="outer", on="column_name")

ID = list(range(1,len(df_dependent_str)+1))
df_dependent_str.insert(0, "ID", ID, True)
df_independent_str.insert(0, "ID", ID, True)
df_dependent_int.insert(0, "ID", ID, True)
df_independent_int.insert(0, "ID", ID, True)

df_total=pd.DataFrame()
df_total_p=pd.DataFrame()

def pearsonspearman(i,ii,dft,type):
    df =pd.DataFrame()
    dfp=pd.DataFrame()
    dfs=pd.DataFrame()
    if type == 'p':
        test_name='pearson'
        pearson_coef,p_value = stats.pearsonr(dft[i],dft[ii])
        if p_value < 0.05:
            d={'dependent':[i],'independent':[ii],'test_name':[test_name],'p_value':[p_value],'pearson_coef':[pearson_coef]}
            dfp=pd.DataFrame(data=d)
    elif type == 's':
        test_name='spearman'
        correlation,p_value = stats.spearmanr(dft[i],dft[ii])
        list2=[i,ii,test_name,p_value,correlation]
        if p_value < 0.05:
            d={'dependent':[i],'independent':[ii],'test_name':[test_name],'p_value':[p_value],'correlation':[correlation]}
            dfp=pd.DataFrame(data=d)
    df = df.append(dfs)
    df = df.append(dfp)
    return df

def chisquare(s,ss):
    df =pd.DataFrame()
    dfc=pd.DataFrame()
    test_name='chi-square'
    df1=df_dependent_str.copy()
    df1.dropna(subset=[s],axis=0,inplace=True)
    df2=df_independent_str.copy()
    df2.dropna(subset=[ss],axis=0,inplace=True)
    dft = pd.merge(df1, df2, how="inner" , on="ID")
    cont_table = pd.crosstab(dft[s], dft[ss])
    chi_val, p_value, dof, expected = scipy.stats.chi2_contingency(cont_table, correction = True)
    if p_value < 0.05:
        d={'dependent':[s],'independent':[ss],'test_name':[test_name],'p_value':[p_value],'chi value':[chi_val],'degrees of freedom':[dof]}
        dfc=pd.DataFrame(data=d)
    df = df.append(dfc)
    return df

def ttestmannwhitneyu(i,ss,list1,p_value_kolmogorovsmirnov,dft):
    df =pd.DataFrame()
    dfc=pd.DataFrame()
    data_list=[]
    for a in list1:
        data = dft.loc[dft[ss] == a,[i]]
        list=data.values.tolist()
        flat_list = []
        for sublist in list:
            for item in sublist:
                flat_list.append(item)
        data_list.append(flat_list)
    if p_value_kolmogorovsmirnov >= 0.05:
        test_name='independent t-test'
        statistic,p_value= stats.ttest_ind(data_list[0],data_list[1])
    else:
        test_name='mann-whitney-u'
        statistic,p_value= scipy.stats.mannwhitneyu(data_list[0],data_list[1])
    if p_value < 0.05:
        d={'dependent':[i],'independent':[ss],'test_name':[test_name],'p_value':[p_value],'statistic':[statistic]}
        dfc=pd.DataFrame(data=d)
    df = df.append(dfc)
    return df

def posthoc(i,ss,dft):
    result=dft.pairwise_tukey(dv=i, between=ss)
    sig = result.loc[result['p-tukey'] < 0.05]
    sig.insert(0, "dependent", i, True)
    sig.insert(1, "independent", ss, True)
    sig.insert(2, "test_name", 'post hoc', True)
    df = sig.rename(columns={'p-tukey': 'p_value'})
    return df

for i in df_dependent_int:
    if i == 'ID':
        continue
    else:
        for ii in df_independent_int:
            if ii == 'ID':
                continue
            else:
                df1=df_dependent_int.copy()
                df1.dropna(subset=[i],axis=0,inplace=True)
                list=df1[i].values.tolist()
                num = np.array(list)
                ksstat,p_value_kolmogorovsmirnov = lilliefors(num)
                df2=df_independent_int.copy()
                df2.dropna(subset=[ii],axis=0,inplace=True)
                list=df2[ii].values.tolist()
                num = np.array(list)
                ksstat,p_value_kolmogorovsmirnov_2 = lilliefors(num)
                dft = pd.merge(df1, df2, how="inner" , on="ID")
                if p_value_kolmogorovsmirnov < 0.05 and p_value_kolmogorovsmirnov_2 < 0.05:
                    type='s'
                else:
                    type='p'
                l = pearsonspearman(i,ii,dft,type)
                df_total=df_total.append(l)

for s in df_dependent_str:
    if s == 'ID':
        continue
    else:
        for ss in df_independent_str:
            if ss == 'ID':
                continue
            else:
                l = chisquare(s,ss)
                df_total=df_total.append(l)

for i in df_dependent_int:
    if i == 'ID':
        continue
    else:
        for ss in df_independent_str:
            if ss == 'ID':
                continue
            else:
                df1=df_dependent_int.copy()
                df1.dropna(subset=[i],axis=0,inplace=True)
                list=df1[i].values.tolist()
                num = np.array(list)
                ksstat,p_value_kolmogorovsmirnov = lilliefors(num)
                df2=df_independent_str.copy()
                df2.dropna(subset=[ss],axis=0,inplace=True)
                dft = pd.merge(df1, df2, how="inner" , on="ID")
                df_independent_str_2=df2.copy()
                products_list = dft[ss].values.tolist()
                list1 = set(products_list)
                if len(list1) == 2:
                    l = ttestmannwhitneyu(i,ss,list1,p_value_kolmogorovsmirnov,dft)
                    df_total=df_total.append(l)
                elif len(list1) > 2:
                    l = posthoc(i,ss,dft)
                    df_total_p=df_total_p.append(l)

for i in df_independent_int:
    if i == 'ID':
        continue
    else:
        for ss in df_dependent_str:
            if ss == 'ID':
                continue
            else:
                df1=df_independent_int.copy()
                df1.dropna(subset=[i],axis=0,inplace=True)
                list=df1[i].values.tolist()
                num = np.array(list)
                ksstat,p_value_kolmogorovsmirnov = lilliefors(num)
                df2=df_dependent_str.copy()
                df2.dropna(subset=[ss],axis=0,inplace=True)
                dft = pd.merge(df1, df2, how="inner" , on="ID")
                df_dependent_str_2=df2.copy()
                products_list = dft[ss].values.tolist()
                list1 = set(products_list)
                if len(list1) == 2:
                    l = ttestmannwhitneyu(i,ss,list1,p_value_kolmogorovsmirnov,dft)
                    df_total=df_total.append(l)
                elif len(list1) > 2:
                    l = posthoc(i,ss,dft)
                    df_total_p=df_total_p.append(l)

print('total number of',df_total.shape[0],'significant values was calculated other than post hoc.')
df_total.to_excel("Analytical.xlsx")
print('total number of',df_total_p.shape[0],'significant values was calculated using post hoc.')
df_total_p.to_excel("Analytical-post-hoc.xlsx")
print('total number of',df_total_d.shape[0],'columns were descripted.')
df_total_d.to_excel("Descriptive.xlsx")
