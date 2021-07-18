import pandas as pd

transactions_data = pd.read_csv('transactions.csv', sep=';', na_values='n/a')
clients_data = pd.read_csv('clients.csv', sep=',')

transactions_data['tran_dttm'] = pd.to_datetime(transactions_data['tran_dttm'])
transactions_data['merchant_category'] = transactions_data['merchant_category'].replace(
    ['Продукты', 'товары повседневного спроса'],
    'Продукты и товары повседневного спроса')
transactions_data = transactions_data[
    transactions_data['merchant_category'] != 'Продукты и товары повседневного спроса']

# One-hot encoding
one_hot = pd.get_dummies(transactions_data['merchant_category'])
transactions_data = transactions_data.drop('merchant_category', axis=1)
transactions_data = transactions_data.join(one_hot)

# Remove irrelevant information
transactions_data = transactions_data.drop(
    ['brand', 'credit_tran_amt', 'deposit_tran_amt', 'is_nfc_pay',
     'is_action',
     'term', 'brand_rk'], axis=1)

clients_groups = transactions_data.groupby('client_id')
clients_data = clients_data[clients_data['client_id'].isin(clients_groups.groups)]

purchases_frequency = {}
purchases_amount = {}
for col_name in ['Заправки']:
    # for col_name in one_hot.columns:
    pfs = clients_groups[col_name].apply(sum)
    purchases_frequency[col_name] = pfs

    pas = transactions_data[transactions_data[col_name] == 1].groupby('client_id')['tran_amt'].apply(sum)
    purchases_amount[col_name] = pas

for col_name in ['Заправки']:
    # for col_name in one_hot.columns:
    temp_df = pd.concat([purchases_frequency[col_name], purchases_amount[col_name]], axis=1)
    temp_df.columns = [col_name + ' frequency', col_name + ' amount']
    temp_df = temp_df.fillna(0)
    clients_data = pd.merge(clients_data, temp_df, on='client_id')

clients_data.to_csv('extended_clients_data.csv')
