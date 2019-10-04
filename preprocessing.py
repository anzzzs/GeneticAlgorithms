def convert_age(age):
    if   age < 18: return 1
    elif age < 30 and age >= 18: return 2
    elif age < 50 and age >= 30: return 3
    elif age < 70 and age >= 50: return 4
    else: return 5

def preproc_func(data):
    temp = data.copy()
    
    #temp = temp[temp['gender']!=1]
    temp['age_cat']   = temp['age'].apply(convert_age)
    
    temp['sex_male'] = temp['sex'].apply(lambda x: 1 if x==1 else 0)
    
    temp['cp_0'] = temp['cp'].apply(lambda x: 1 if x==0 else 0)
    temp['cp_1'] = temp['cp'].apply(lambda x: 1 if x==1 else 0)
    temp['cp_2'] = temp['cp'].apply(lambda x: 1 if x==2 else 0)
    
    temp['fbs_1'] = temp['fbs'].apply(lambda x: 1 if x==1 else 0)
    
    temp['restecg_0'] = temp['restecg'].apply(lambda x: 1 if x==0 else 0)
    temp['restecg_1'] = temp['restecg'].apply(lambda x: 1 if x==1 else 0)
    
    temp['exang_1'] = temp['exang'].apply(lambda x: 1 if x==1 else 0)
    
    temp['slope_1'] = temp['slope'].apply(lambda x: 1 if x==1 else 0)
    temp['slope_2'] = temp['slope'].apply(lambda x: 1 if x==2 else 0)
    
    temp['ca_0'] = temp['ca'].apply(lambda x: 1 if x==0 else 0)
    temp['ca_1'] = temp['ca'].apply(lambda x: 1 if x==1 else 0)
    temp['ca_2'] = temp['ca'].apply(lambda x: 1 if x==2 else 0)
    temp['ca_3'] = temp['ca'].apply(lambda x: 1 if x==3 else 0)
    
    temp['thal_1'] = temp['thal'].apply(lambda x: 1 if x==1 else 0)
    temp['thal_2'] = temp['thal'].apply(lambda x: 1 if x==2 else 0)
    temp['thal_3'] = temp['thal'].apply(lambda x: 1 if x==3 else 0)

    cols_to_drop = ['sex', 'cp', 'fbs','restecg', 'exang', 'slope', 'ca', 'thal']
    
    temp.drop(cols_to_drop, axis=1, inplace=True)
    
    return temp