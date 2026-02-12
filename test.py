import pandas as pd

file_path = r"E:\2023210119贾正鑫\DrivingData_20EVs.xlsx"
df_activity = pd.read_excel(file_path, sheet_name='Activity')
print("Activity 工作表列名列表:")
print(df_activity.columns.tolist())
print("\n前几行数据:")
print(df_activity.head(10))