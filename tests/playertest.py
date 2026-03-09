import pandas as pd

url = "https://www.sports-reference.com/cbb/players/money-williams-1.html"

tables = pd.read_html(url)

print(f"Number of tables found: {len(tables)}")
print("-" * 80)

for i, df in enumerate(tables):
    print(f"TABLE {i}")
    print("COLUMNS:")
    print(df.columns.tolist())
    print()
    print("FIRST 5 ROWS:")
    print(df.head())
    print("-" * 80)