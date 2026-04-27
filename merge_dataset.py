import pandas as pd

old_data = pd.read_csv("C:/machine_learning/phishing_site_urls.csv")
new_data = pd.read_csv("C:/machine_learning/phishing.csv")

# Convert new dataset to match old format
new_data = new_data[["domain", "label"]]
new_data = new_data.rename(columns={
    "domain": "URL",
    "label": "Label"
})

# Convert labels
new_data["Label"] = new_data["Label"].map({
    1: "bad",
    0: "good"
})

# Combine datasets
combined = pd.concat([old_data, new_data], ignore_index=True)

# Remove duplicate URLs
combined = combined.drop_duplicates(subset=["URL"])

# Save combined dataset
combined.to_csv("C:/machine_learning/combined_phishing_dataset.csv", index=False)

# print("Combined dataset saved successfully!")
# print("Total rows:", len(combined))
# print(combined["Label"].value_counts())