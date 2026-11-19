"""
Linh-An Tran
CSE 163
Loads, filters, cleans, and labels the two food datasets used in this
project: Open Food Facts (OFF) and the TempoLife Food Nutrition Dataset.
Every food ends up with the same columns: name, brand, dataset, group
(packaged / generic / unclear), food_category, form, sugar, and salt
(both in grams per 100 g).

Running this file as a script filters the full OFF export (several GB)
down to a small CSV, reading it in chunks so it never sits in memory.
"""
import csv

import numpy as np
import pandas as pd

OFF_RAW_PATH = 'data/en.openfoodfacts.org.products.csv.gz'
OFF_FILTERED_PATH = 'data/off_filtered.csv'
TEMPOLIFE_PATH = 'data/tempolife-foods.csv'
OFF_COLUMNS = ['code', 'product_name', 'brands', 'categories_tags',
               'sugars_100g', 'salt_100g']
CHUNK_SIZE = 200000

# Shared food categories, in priority order. Each entry pairs the English
# name used in this project with the TempoLife category (Estonian) and the
# OFF category tags that mean the same kind of food. An OFF product that
# carries tags for several categories gets the first one listed here, so
# more specific categories come first (e.g. a yogurt dessert tagged as both
# a dessert and a dairy product counts as a dessert, as in TempoLife).
CATEGORY_MAP = [
    ('Breakfast cereals', 'Hommikusöögid', ['en:breakfast-cereals']),
    ('Soups', 'Supid', ['en:soups']),
    ('Sauces & condiments', 'Kastmed ja maitsed',
     ['en:sauces', 'en:condiments']),
    ('Salty snacks & bars', 'Snäkid ja suupisted',
     ['en:salty-snacks', 'en:bars']),
    ('Desserts & sweets', 'Magustoidud ja küpsetised',
     ['en:desserts', 'en:confectioneries', 'en:frozen-desserts']),
    ('Breads & bakery', 'Leivad ja pagarittooted',
     ['en:breads', 'en:biscuits-and-cakes', 'en:crackers']),
    ('Grains & pasta', 'Teravili ja pasta',
     ['en:pastas', 'en:rices', 'en:flours']),
    ('Dairy', 'Piimatooted', ['en:dairies']),
    ('Ready meals', 'Valmistoidud', ['en:meals']),
]

# An ALL-CAPS word of 3+ letters marks a brand in USDA SR Legacy names
# (e.g. "Cereals ready-to-eat, POST, GOLDEN CRISP"). The listed words are
# capitalized but are not brands, so they are skipped.
NOT_BRAND_WORDS = 'USDA|BBQ|NFS|UHT'
BRAND_PATTERN = r"\b(?!(?:" + NOT_BRAND_WORDS + r")\b)[A-Z][A-Z'&\-]{2,}\b"
# Restaurant, fast food, and school lunch items are neither packaged nor
# generic grocery foods. USDA names them either by type or by chain.
RESTAURANT_CHAINS = ("KFC|BURGER KING|POPEYES|DENNY'S|McDONALD'S|WENDY'S|"
                     "APPLEBEE'S|SUBWAY|TACO BELL|CRACKER BARREL|"
                     "T\\.G\\.I\\. FRIDAY'S|CARRABBA'S|OLIVE GARDEN|"
                     "ON THE BORDER|PIZZA HUT|CHICK-FIL-A|DOMINO'S|"
                     "LITTLE CAESARS|PAPA JOHN'S|ARBY'S")
RESTAURANT_PATTERN = (r'^(?:Restaurant|Fast food|School Lunch)|'
                      r'\b(?:' + RESTAURANT_CHAINS + r')')
# Words showing a USDA food was cooked, mixed with water or milk, or
# otherwise changed from the form it is sold in.
PREPARED_PATTERN = (r'\b(?:cooked|prepared|reconstituted|boiled|'
                    r'with water|equal volume)\b')


def off_tag_pattern(tags: list[str]) -> str:
    """
    Takes a list of OFF category tags and returns a regular expression that
    matches a comma-separated categories_tags string containing any of
    those tags as a whole tag (so 'en:breads' does not match
    'en:breadsticks').
    """
    joined = '|'.join(tags)
    return r'(?:^|,)(?:' + joined + r')(?:,|$)'


def all_off_tags_pattern() -> str:
    """
    Returns a regular expression matching any OFF tag in CATEGORY_MAP.
    """
    all_tags = [tag for _, _, tags in CATEGORY_MAP for tag in tags]
    return off_tag_pattern(all_tags)


def filter_off(raw_path: str, out_path: str,
               chunk_size: int = CHUNK_SIZE) -> dict[str, int]:
    """
    Reads the OFF export at raw_path (tab-separated, may be gzipped) in
    chunks of chunk_size rows, keeps only OFF_COLUMNS and only the products
    tagged with a category in CATEGORY_MAP, and writes them to out_path as
    a CSV. Malformed lines (whose barcode is not all digits, a sign the
    fields have shifted) are dropped. Returns a dictionary with the number
    of rows read ('rows_read'), dropped as malformed ('rows_malformed'),
    and kept ('rows_kept').
    """
    pattern = all_off_tags_pattern()
    reader = pd.read_csv(raw_path, sep='\t', usecols=OFF_COLUMNS,
                         dtype=str, chunksize=chunk_size,
                         quoting=csv.QUOTE_NONE, on_bad_lines='skip',
                         encoding='utf-8', encoding_errors='replace')
    rows_read = 0
    rows_malformed = 0
    kept_chunks = []
    for chunk in reader:
        rows_read += len(chunk)
        # With usecols, pandas keeps lines that have too many fields
        # instead of skipping them, so check the barcode as well.
        valid_code = chunk['code'].fillna('').str.fullmatch(r'\d+')
        rows_malformed += int((~valid_code).sum())
        tags = chunk['categories_tags'].fillna('')
        in_project = tags.str.contains(pattern, regex=True)
        kept_chunks.append(chunk[valid_code & in_project])
    kept = pd.concat(kept_chunks, ignore_index=True)
    kept.to_csv(out_path, index=False)
    return {'rows_read': rows_read, 'rows_malformed': rows_malformed,
            'rows_kept': len(kept)}


def clean_nutrients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with 'sugar' and 'salt' columns and returns a copy in
    which both are numeric. Text that is not a number, negative values, and
    values above 100 g per 100 g become missing (NaN), since they cannot be
    real amounts. The input is not modified.
    """
    result = df.copy()
    for column in ['sugar', 'salt']:
        values = pd.to_numeric(result[column], errors='coerce')
        impossible = (values < 0) | (values > 100)
        result[column] = values.mask(impossible)
    return result


def count_impossible(df: pd.DataFrame, column: str) -> int:
    """
    Takes a DataFrame and a column name and returns how many values in that
    column are numbers below 0 or above 100 (impossible per 100 g).
    """
    values = pd.to_numeric(df[column], errors='coerce')
    return int(((values < 0) | (values > 100)).sum())


def label_tempolife_group(df: pd.DataFrame) -> pd.Series:
    """
    Takes TempoLife rows (columns 'name' and 'source') and returns a Series
    labelling each row 'packaged', 'generic', or 'unclear':
    - rows with no source (curated by TempoLife from a mix of labels and
      tables) and restaurant, fast food, and school lunch rows are
      'unclear';
    - USDA rows whose name contains an ALL-CAPS brand word are 'packaged';
    - all other USDA rows are 'generic'.
    """
    is_usda = df['source'] == 'usda_sr_legacy'
    is_restaurant = df['name'].str.contains(RESTAURANT_PATTERN, case=False,
                                            regex=True)
    has_brand = df['name'].str.contains(BRAND_PATTERN, regex=True)
    labels = np.select([~is_usda | is_restaurant, has_brand],
                       ['unclear', 'packaged'], default='generic')
    return pd.Series(labels, index=df.index)


def label_form(names: pd.Series) -> pd.Series:
    """
    Takes a Series of food names and returns a Series that is 'prepared'
    for foods described as cooked or mixed with water / milk and 'as sold'
    otherwise.
    """
    is_prepared = names.str.contains(PREPARED_PATTERN, case=False,
                                     regex=True)
    return pd.Series(np.where(is_prepared, 'prepared', 'as sold'),
                     index=names.index)


def load_tempolife(path: str) -> pd.DataFrame:
    """
    Loads the TempoLife CSV at path and returns it with the shared project
    columns. food_category is the English name from CATEGORY_MAP, or NaN
    for TempoLife categories outside the project. Rows with no source list
    sugar as 0 for every food (even jam), so their sugar is treated as
    missing. Sugar and salt are cleaned with clean_nutrients.
    """
    raw = pd.read_csv(path)
    placeholder_sugar = raw['source'].isna()
    category_names = {tl: name for name, tl, _ in CATEGORY_MAP}
    result = pd.DataFrame({
        'name': raw['name'],
        'brand': raw['brand'],
        'dataset': 'TempoLife',
        'source': raw['source'],
        'tempolife_category': raw['category'],
        'food_category': raw['category'].map(category_names),
        'sugar': raw['sugar'].mask(placeholder_sugar),
        'salt': raw['salt'],
    })
    result['group'] = label_tempolife_group(result)
    result['form'] = label_form(result['name'])
    return clean_nutrients(result)


def assign_off_category(tags: pd.Series) -> pd.Series:
    """
    Takes a Series of OFF categories_tags strings and returns the English
    food category for each, using the first matching entry in
    CATEGORY_MAP. Products matching no entry get NaN.
    """
    tags = tags.fillna('')
    conditions = [tags.str.contains(off_tag_pattern(off_tags), regex=True)
                  for _, _, off_tags in CATEGORY_MAP]
    names = [name for name, _, _ in CATEGORY_MAP]
    categories = np.select(conditions, names, default='')
    return pd.Series(categories, index=tags.index).replace('', np.nan)


def load_off(path: str) -> pd.DataFrame:
    """
    Loads the filtered OFF CSV at path and returns it with the shared
    project columns. A product is 'packaged' if its brands field is filled
    in and 'unclear' otherwise. Every OFF product is treated as sold
    as-is. Sugar and salt are cleaned with clean_nutrients.
    """
    raw = pd.read_csv(path, dtype={'code': str})
    brands = raw['brands'].fillna('').str.strip()
    result = pd.DataFrame({
        'name': raw['product_name'],
        'brand': brands.replace('', np.nan),
        'dataset': 'Open Food Facts',
        'source': 'open_food_facts',
        'categories_tags': raw['categories_tags'],
        'food_category': assign_off_category(raw['categories_tags']),
        'sugar': raw['sugars_100g'],
        'salt': raw['salt_100g'],
    })
    result['group'] = np.where(brands != '', 'packaged', 'unclear')
    result['form'] = 'as sold'
    return clean_nutrients(result)


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of foods and returns a copy without exact duplicates:
    rows with the same dataset, name, brand, sugar, and salt as an earlier
    row. Rows that share a name but differ in sugar or salt are kept,
    because TempoLife cuts long names short and different foods (e.g.
    salted and unsalted versions) can end up with the same name.
    """
    key = ['dataset', 'name', 'brand', 'sugar', 'salt']
    return df.drop_duplicates(subset=key).reset_index(drop=True)


def analysis_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of labelled foods and returns only the rows used in
    the comparisons: in a project food category, labelled packaged or
    generic, sold as-is, and not an exact duplicate.
    """
    in_category = df['food_category'].notna()
    clear_group = df['group'].isin(['packaged', 'generic'])
    as_sold = df['form'] == 'as sold'
    return drop_exact_duplicates(df[in_category & clear_group & as_sold])


def main():
    counts = filter_off(OFF_RAW_PATH, OFF_FILTERED_PATH)
    print('Open Food Facts rows read:', counts['rows_read'])
    print('Malformed rows dropped:', counts['rows_malformed'])
    print('Rows kept in project categories:', counts['rows_kept'])
    print('Saved to', OFF_FILTERED_PATH)


if __name__ == '__main__':
    main()
