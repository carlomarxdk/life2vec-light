import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class UserRecordsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, user_base: pd.DataFrame, vocab: pd.DataFrame):
        """
        Initializes the dataset with the dataframe.
        Groups the dataframe by 'UserID' for efficient access.
        """
        self.dataframe = dataframe
        self.user_base = user_base
        self.user_records = dataframe.groupby('user_id')

    def __len__(self):
        """
        Returns the number of unique users in the dataset.
        """
        return self.user_base.shape[0]

    def __getitem__(self, idx):
        """
        Returns all records for the user corresponding to the index `idx`.
        """
        user_id = self.user_ids[idx]
        user_records = self.user_groups.get_group(user_id)

        # Convert the DataFrame to a PyTorch tensor.
        # Here you might need to adjust the conversion based on your dataframe's structure and what you want to predict.
        # The example below assumes you want to convert all columns except 'UserID' to tensors.
        # Adjust the column names and types as needed for your specific case.

        # Example of converting 'Date of Record' and 'Birthday' to ordinal and then to tensors,
        # and mapping 'Sex' to a binary representation.
        user_records['Date of Record Ordinal'] = pd.to_datetime(
            user_records['Date of Record']).apply(lambda x: x.toordinal())
        user_records['Birthday Ordinal'] = pd.to_datetime(
            user_records['Birthday']).apply(lambda x: x.toordinal())
        user_records['Sex'] = user_records['Sex'].apply(
            lambda x: 0 if x == 'Male' else 1)

        # Converting to tensor. Adjust as per your dataframe's needs.
        features = torch.tensor(user_records[[
                                'Date of Record Ordinal', 'Birthday Ordinal', 'Sex']].values, dtype=torch.float)
        targets = torch.tensor(user_records['Salary'].values, dtype=torch.float).unsqueeze(
            1)  # Assuming 'Salary' is the target

        return features, targets


# Assuming df_records is your DataFrame
dataset = CompleteUserRecordsDataset(df_records)

# Example usage
for i in range(len(dataset)):
    features, targets = dataset[i]
    print(f"User {i+1} features:\n{features}\nTargets:\n{targets}\n")
    # Break after first to avoid flooding the output, remove this in real usage
    if i == 0:
        break

# To use DataLoader for batching, note that you might need a custom collate function
# if each user has a different number of records
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
