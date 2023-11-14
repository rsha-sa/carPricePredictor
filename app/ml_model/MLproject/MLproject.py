
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

plt.style.use('ggplot')


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the CSV file
file_path = os.path.join(current_dir, 'Real-life-example.csv')

# Read the CSV file
df = pd.read_csv(file_path)


df.info()

df.index

df.columns

df.describe(include='all')

df['Brand'].value_counts()

df['Engine Type'].value_counts()

df['Price'].value_counts()



df.isna().sum()

df2 = df.dropna(axis=0)

df2.isna().sum()

print(df2.duplicated().sum())
df2[df2.duplicated()]

# df2 = df2.drop_duplicates(keep ='first')
# df2.duplicated().sum()


plt.figure(figsize=(6,5))
sns.countplot(y=df2.Brand, hue =df2.Registration)
plt.show()

plt.figure(figsize=(4,4))
sns.countplot(x=df2["Engine Type"])
plt.show()

sns.distplot(df2['Price'])

plt.figure(figsize=(10,5))
df2.groupby('Year')['Price'].mean().plot(kind = 'bar')
plt.show()

# the relation beteen price and year

f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize =(15,3))
ax1.scatter(x = "Year" , y = "Price",data = df2)
ax1.set_title("price and Year")
ax2.scatter(x=df2.Mileage , y= df2.Price)
ax2.set_title("Price and Mileage")
ax3.scatter(df2['EngineV'] , df2['Price'])
ax3.set_title("Price and EngineV")
plt.show()


brands_to_encode = ['Volkswagen', 'Mercedes-Benz', 'BMW', 'Toyota', 'Renault', 'Audi', 'Mitsubishi']

brands_encoded = pd.get_dummies(df2['Brand'].apply(lambda x: x if x in brands_to_encode else 'Other'), prefix='Brand')
brands_encoded =brands_encoded.astype(int)

df2 = pd.concat([df2, brands_encoded], axis=1)

df2 = df2.drop('Brand', axis=1)

# Display the updated DataFrame
print(df2)


EngineType_encode = ['Diesel', 'Petrol', 'Gas', 'other']

# Apply One-Hot Encoding
EngineType_encode = pd.get_dummies(df2['Engine Type'].apply(lambda x: x if x in EngineType_encode else 'Other'), prefix='Engine Type')
EngineType_encode = EngineType_encode.astype(int)

df2 = pd.concat([df2, EngineType_encode], axis=1)

df2 = df2.drop('Engine Type', axis=1)

# Display the updated DataFrame
print(df2)


current_year = datetime.datetime.now().year

df2['Age'] = current_year - df2['Year']

df2 = df2.drop('Year', axis=1)

# Display the updated DataFrame
print(df2)


print('before handling the outliers\n\n', df2['EngineV'].describe())

upper_bound = df2['EngineV'].quantile(0.9949)

# Identify outliers based on the threshold
outliers = df2[df2['EngineV'] > upper_bound]

print(len(outliers))

print("\n-------------------------\n")

df2.drop(outliers.index, inplace=True)

print('After handling the outliers\n\n', df2['EngineV'].describe())



columns_to_drop = ['Model', 'Body', 'Registration']
df2 = df2.drop(columns=columns_to_drop)

# Display the updated DataFrame
print(df2)



numerical_features = ['Price', 'Mileage', 'EngineV', 'Age']

# Create a StandardScaler object
scaler = StandardScaler()

df2[numerical_features] = scaler.fit_transform(df2[numerical_features])


y = df2['Price'].values.reshape(-1, 1)
x = df2.drop(['Price'], axis = 1).values
x,y


# Split the data using train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Convert NumPy arrays to PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.FloatTensor(y_train)
x_test_tensor = torch.FloatTensor(x_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create datasets and loaders
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# Create DataLoaders for the train and test sets
# Note, that we shuffle only training data, not testing.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Check if a GPU (CUDA) is available, and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class CarPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(CarPricePredictor, self).__init__()

        self.layer1 = nn.Linear(input_size, 420)
        self.batch_norm1 = nn.BatchNorm1d(420)
        self.dropout1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(420, 170)
        self.batch_norm2 = nn.BatchNorm1d(170)
        self.dropout2 = nn.Dropout(0.3)

        self.layer3 = nn.Linear(170, 90)
        self.batch_norm3 = nn.BatchNorm1d(90)
        self.dropout3 = nn.Dropout(0.3)

        self.output_layer = nn.Linear(90, 1)

    def forward(self, x):
        x = nn.functional.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout1(x)

        x = nn.functional.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout2(x)

        x = nn.functional.relu(self.batch_norm3(self.layer3(x)))
        x = self.dropout3(x)

        x = self.output_layer(x)
        return x

def rms(y, yhat):
    return np.sqrt(np.mean(np.square(y - yhat)))

def linear_regression(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

theta = linear_regression(x_train, y_train)
e = rms(y_train, x_train @ theta)
et = rms(y_test, x_test @ theta)
print(e, et)


input_size = x_train.shape[1]
model = CarPricePredictor(input_size)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



x_train_tensor = x_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
x_test_tensor = x_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

model = model.to(device)


# Train the model
num_epochs =  50
loss_values = []  # to store loss values

for epoch in range(num_epochs):

    model.train()

    # Forward pass
    yhat = model(x_train_tensor)
    loss = loss_fn(yhat, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())


    # Print training progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

print("Training complete.")

# Plot the training losses

plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss', color='Red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.show()


#Testing the model's preformance

#Evaluate the model on the test set

with torch.no_grad():

    model.eval()

    yhat_train = model(x_train_tensor).squeeze()

    e = rms(yhat_train.cpu().numpy(), y_train_tensor.cpu().numpy())

    yhat_test = model(x_test_tensor).squeeze()

    et = rms(yhat_test.cpu().numpy(), y_test_tensor.cpu().numpy())

    r2 = r2_score(yhat_train.cpu().numpy(), y_train_tensor.cpu().numpy())
    r2t = r2_score(yhat_test.cpu().numpy(), y_test_tensor.cpu().numpy())

# Print the results
print(f'RMS (Training): {e:.4f}\n')
print(f'RMS (Testing): {et:.4f}\n')
print('--------------------------------------')
print(f'R-squared (Training): {r2:.4f}\n')
print(f'R-squared (Testing): {r2t:.4f}\n')

# Set the model back to training mode
model.train()

torch.save(model.state_dict(), 'CarPricePredictor.pth')


