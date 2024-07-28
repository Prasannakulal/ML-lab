import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
customers_df = pd.read_csv("3c.csv")
orders_df = pd.read_csv("3o.csv")
products_df = pd.read_csv("3p.csv")
customers_df['age'] = customers_df['age'].fillna(customers_df['age'].mean())
customers_df['email'] = customers_df['email'].fillna('N/A')
merged_df = pd.merge(customers_df, orders_df, on='customer_id')
merged_df = pd.merge(merged_df, products_df, on='product_id')
merged_df['total_price'] = merged_df['quantity'] * merged_df['price']
merged_df['Feed_back'] = np.where(merged_df['quantity'] > 1, "Good", "Bad")
print("Cleaned, Integrated, and Transformed Data:")
print(merged_df)
ordinal_encoder = OrdinalEncoder()
label_encoder = LabelEncoder()
merged_ordinal_encoded = ordinal_encoder.fit_transform(merged_df.drop(columns=['Feed_back']))
feed_back_encoded = label_encoder.fit_transform(merged_df['Feed_back'])
print("Features \n", merged_ordinal_encoded)
print("Target \n", feed_back_encoded)
