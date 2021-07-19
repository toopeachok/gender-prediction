import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

transactions_data = pd.read_csv('transactions.csv', sep=';', na_values='n/a')
clients_data = pd.read_csv('clients.csv', sep=',')

transactions_data['merchant_category'] = transactions_data['merchant_category'].replace(
    ['Продукты', 'товары повседневного спроса'],
    'Продукты и товары повседневного спроса')
transactions_data = transactions_data[
    transactions_data['merchant_category'] != 'Продукты и товары повседневного спроса']

# One-hot encoding
one_hot = pd.get_dummies(transactions_data['merchant_category'])
transactions_data = transactions_data.drop('merchant_category', axis=1)
transactions_data = transactions_data.join(one_hot)

clients_groups = transactions_data.groupby('client_id')
clients_data = clients_data[clients_data['client_id'].isin(clients_groups.groups)]

purchases_frequency = {}
purchases_amount = {}
for col_name in one_hot.columns:
    pfs = clients_groups[col_name].apply(sum)
    purchases_frequency[col_name] = pfs

    pas = transactions_data[transactions_data[col_name] == 1].groupby('client_id')['tran_amt'].apply(sum)
    purchases_amount[col_name] = pas

for col_name in one_hot.columns:
    temp_df = pd.concat([purchases_frequency[col_name], purchases_amount[col_name]], axis=1)
    temp_df.columns = [col_name + ' frequency', col_name + ' amount']
    temp_df = temp_df.fillna(0)
    clients_data = pd.merge(clients_data, temp_df, on='client_id')

clients_data.to_csv('extended_clients_data.csv')

clients_data = clients_data.drop(
    ['credit_limit_initial', 'add_product', 'mp_first_login', 'mp_last_login', 'tp_num',
     'city_population_category', 'app_channel'], axis=1)
clients_data.set_index('client_id', inplace=True)
clients_data = clients_data[clients_data['gender_id'] != -1]

train_set, test_set = train_test_split(clients_data, test_size=0.3, random_state=42)
train_set_labels = train_set['gender_id'].copy() == 2
train_set = train_set.drop('gender_id', axis=1)
cat_attributes = ['education_id', 'marital_status_id', 'employment_status_cd']
num_attributes = [x for x in train_set.columns if x not in cat_attributes]
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', OneHotEncoder(), cat_attributes),
])
train_set_fit = full_pipeline.fit_transform(train_set)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(train_set_fit, train_set_labels)
sgd_clf.predict(train_set_fit)
