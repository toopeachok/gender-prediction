import pandas as pd

transactions_data = pd.read_csv('transactions.csv', sep=';')

transactions_data['tran_dttm'] = pd.to_datetime(transactions_data['tran_dttm'])
transactions_data['merchant_category'] = transactions_data['merchant_category'].replace(
    ['Продукты', 'товары повседневного спроса'],
    'Продукты и товары повседневного спроса')

# One-hot encoding
one_hot = pd.get_dummies(transactions_data['merchant_category'])
transactions_data = transactions_data.drop('merchant_category', axis=1)
transactions_data = transactions_data.join(one_hot)

# Remove irrelevant information
transactions_data = transactions_data[transactions_data['Продукты и товары повседневного спроса'] == 0]
transactions_data = transactions_data.drop(
    ['Продукты и товары повседневного спроса', 'brand', 'credit_tran_amt', 'deposit_tran_amt', 'is_nfc_pay',
     'is_action',
     'term', 'brand_rk'], axis=1)
one_hot = one_hot.drop('Продукты и товары повседневного спроса', axis=1)

clients_groups = transactions_data.groupby('client_id')
gas_stations_info = transactions_data[transactions_data['Заправки'] == 1].groupby('client_id')
print(gas_stations_info['tran_amt'].apply(sum))
print(clients_groups['Заправки'].apply(sum))
print(clients_groups.tran_amt.sum())
print(transactions_data)
