import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

import os
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import joblib
import datetime

plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

#reading the file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'Real-life-example.csv')
df = pd.read_csv(file_path)



# Data Pre-processing

# handling null values

df2 = df.dropna(axis=0)

# Handling duplicated values

df2 = df2.drop_duplicates(keep ='first')

df2 = df2[['Brand', 'Price', 'Body', 'Mileage', 'EngineV', 'Year', 'Engine Type', 'Registration', 'Model']]

#transforming the year attribut to better representation

current_year = datetime.datetime.now().year

df2['Age'] = current_year - df2['Year']

df2 = df2.drop('Year', axis=1)


#rearrange
df2 = df2[['Brand', 'Price', 'Body', 'Mileage', 'EngineV', 'Age', 'Engine Type', 'Registration', 'Model']]


# Apply One-Hot Encoding for engine type

EngineType_to_encode = ['Diesel', 'Petrol', 'Gas', 'other']


EngineType_to_encode = pd.get_dummies(df2['Engine Type'].apply(lambda x: x if x in EngineType_to_encode else 'Other'), prefix='Engine Type')
EngineType_to_encode = EngineType_to_encode.astype(int)

df2 = pd.concat([df2, EngineType_to_encode], axis=1)
df2 = df2.drop('Engine Type', axis=1)

# applying one-hot encoding for brand_model

unique_brands = ['Volkswagen', 'Mercedes-Benz', 'BMW', 'Toyota', 'Renault', 'Audi', 'Mitsubishi']

for current_brand in unique_brands:
    brand_data = df2[df2['Brand'] == current_brand]

    #encoding the top 10 models for each brand
    top_models = brand_data['Model'].value_counts().nlargest(10).index

    for top_model in top_models:
        df2[f'{current_brand}_{top_model}'] = (df2['Brand'] == current_brand) & (df2['Model'] == top_model)

    df2[f'{current_brand}_Other'] = (df2['Brand'] == current_brand) & (~df2['Model'].isin(top_models))

df2 = df2.drop(['Brand', 'Model'], axis=1)

boolean_columns = df2.columns[df2.dtypes == bool]
df2[boolean_columns] = df2[boolean_columns].astype(int)


# Handling the outliers of engineV


upper_bound = df2['EngineV'].quantile(0.9949)

outliers = df2[df2['EngineV'] > upper_bound]

df2.drop(outliers.index, inplace=True)

# Handling the outliers of price

Q1 = df2['Price'].quantile(0.25)
Q3 = df2['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df2[(df2['Price'] < lower_bound) | (df2['Price'] > upper_bound)]

df2.drop(outliers.index, inplace=True)


# Feature Selection

columns_to_drop = ['Body', 'Registration']
df2 = df2.drop(columns=columns_to_drop)

# scaling the numerical features

numerical_features = ['Mileage', 'EngineV', 'Age']

scaler = StandardScaler()

df2[numerical_features] = scaler.fit_transform(df2[numerical_features])


# Model Building


y = df2['Price'].values.reshape(-1, 1)
x = df2.drop(['Price'], axis = 1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.FloatTensor(y_train)
x_test_tensor = torch.FloatTensor(x_test)
y_test_tensor = torch.FloatTensor(y_test)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Defining the neural network class

class CarPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(CarPricePredictor, self).__init__()

        self.layer1 = nn.Linear(input_size, 300)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(300, 120)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(120, 60)
        self.output_layer = nn.Linear(60, 1)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.layer2(x))
        x = self.dropout2(x)
        x = nn.functional.relu(self.layer3(x))
        x = self.output_layer(x)
        return x
    
    

# Evaluate Function 

def evaluate_model(true, predicted):

    true = true.cpu().detach().numpy()
    predicted = predicted.cpu().detach().numpy()

    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)

    return mae, rmse, r2_square



# Initializing the model, loss_fn, optimizer


input_size = x_train.shape[1]
model = CarPricePredictor(input_size)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

x_train_tensor = x_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

model = model.to(device)


# start training the model


num_epochs =  250
loss_values = []  

for epoch in range(num_epochs):

    model.train()

    yhat = model(x_train_tensor)
    loss = loss_fn(yhat, y_train_tensor)

    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train_tensor, yhat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


print ('---------------------------------')

print('Model performance for Training set\n')
print("- Root Mean Squared Error: {:.4f}\n".format(model_train_rmse))
print("- Mean Absolute Error: {:.4f}\n".format(model_train_mae))
print("- R2 Score: {:.4f}\n".format(model_train_r2))

print ('---------------------------------')


print("Training complete.")

# Plot the training losses

plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss', color='Red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.show()


# testing the model

with torch.no_grad():

    model.eval()

    yhat_train = model(x_train_tensor).squeeze()

    yhat_test = model(x_test_tensor).squeeze()

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test_tensor, yhat_test)

    r2 = r2_score(yhat_train.cpu().numpy(), y_train_tensor.cpu().numpy())
    r2t = r2_score(yhat_test.cpu().numpy(), y_test_tensor.cpu().numpy())


print('Model performance for Test set\n')
print("- Root Mean Squared Error: {:.4f}\n".format(model_test_rmse))
print("- Mean Absolute Error: {:.4f}\n".format(model_test_mae))
print("- R2 Score: {:.4f}\n".format(model_test_r2))
print('--------------------------------------')


print('--------------------------------------')

print(f'R-squared (Training): {r2:.4f}\n')
print(f'R-squared (Testing): {r2t:.4f}\n')

model.train()

joblib.dump(scaler, 'scaler.joblib')

torch.save(model.state_dict(), 'CarPricePredictor.pth')