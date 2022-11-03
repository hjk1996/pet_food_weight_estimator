from dataset import load_meta_data

if __name__ == "__main__":
    df = load_meta_data("./data/image_meta_data.csv")
    # df = df[['bowl_type', 'food_type', 'gram', 'image_name']]
    # df.to_csv('./image_meta_data.csv', index=False)
    hash = (
        "bt"
        + df["bowl_type"].astype(str)
        + "ft"
        + df["food_type"].astype(str)
        + "gram"
        + df["gram"].astype(str)
    )
    print(hash.head())
