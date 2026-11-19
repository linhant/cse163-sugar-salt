"""
Linh-An Tran
CSE 163
Tests for data_processing.py and eda.py. The tests use small files in
test_data/ (an Open Food Facts style file with a shifted line, a
non-number, an impossible value, and overlapping tags, and a 9-row
TempoLife sample) and small hand-made DataFrames whose answers were worked
out by hand.
"""
import os

import numpy as np
import pandas as pd

import data_processing as dp
import eda
from cse163_utils import assert_equals

OFF_SAMPLE = 'test_data/off_sample.tsv.gz'
OFF_SAMPLE_FILTERED = 'test_data/off_sample_filtered.csv'
TEMPOLIFE_SAMPLE = 'test_data/tempolife_sample.csv'


def test_off_tag_pattern() -> None:
    """
    Tests off_tag_pattern: whole tags match anywhere in the list, partial
    tags do not.
    """
    pattern = dp.off_tag_pattern(['en:breads', 'en:soups'])
    tags = pd.Series(['en:breads', 'en:plant-based-foods,en:breads',
                      'en:soups,en:meals', 'en:breadsticks',
                      'en:sourdough-breads', ''])
    assert_equals([True, True, True, False, False, False],
                  tags.str.contains(pattern, regex=True).tolist())


def test_filter_off() -> None:
    """
    Tests filter_off on the sample file, using a chunk size of 3 so that
    the chunked reading is exercised.
    """
    counts = dp.filter_off(OFF_SAMPLE, OFF_SAMPLE_FILTERED, chunk_size=3)
    # pandas reads the shifted line (too many fields) as a 9th row; it is
    # tagged en:breads but has a non-digit barcode, so it is dropped
    assert_equals({'rows_read': 9, 'rows_malformed': 1, 'rows_kept': 6},
                  counts)
    kept = pd.read_csv(OFF_SAMPLE_FILTERED, dtype={'code': str})
    assert_equals(dp.OFF_COLUMNS, list(kept.columns))
    # breadsticks (004) and cola (005) are outside the project categories
    assert_equals(['001', '002', '003', '006', '007', '008'],
                  kept['code'].tolist())


def test_clean_nutrients() -> None:
    """
    Tests clean_nutrients: text, negative values, and values above 100
    become NaN, valid values (including 0 and 100) are kept, and the input
    is not modified.
    """
    raw = pd.DataFrame({'sugar': ['5', 'abc', '-1', '100'],
                        'salt': [0, 150, 2.5, None]})
    cleaned = dp.clean_nutrients(raw)
    expected = pd.DataFrame({'sugar': [5.0, np.nan, np.nan, 100.0],
                             'salt': [0.0, np.nan, 2.5, np.nan]})
    assert_equals(expected, cleaned)
    assert_equals('abc', raw.loc[1, 'sugar'])


def test_count_impossible() -> None:
    """
    Tests count_impossible.
    """
    df = pd.DataFrame({'salt': [-0.5, 0, 50, 100, 100.1, None, 'x']})
    assert_equals(2, dp.count_impossible(df, 'salt'))
    assert_equals(0, dp.count_impossible(pd.DataFrame({'salt': [1]}),
                                         'salt'))


def test_label_tempolife_group() -> None:
    """
    Tests label_tempolife_group on each kind of row.
    """
    usda = 'usda_sr_legacy'
    df = pd.DataFrame({
        'name': ['Cereals ready-to-eat, POST, GOLDEN CRISP',
                 'Cereals, oats, regular, dry',
                 'Beef, t-bone steak, USDA choice, cooked',
                 'Chicken, rotisserie, BBQ, breast',
                 "McDONALD'S, french fries",
                 'Restaurant, Chinese, lemon chicken',
                 'Berry jam',
                 'Yogurt, Greek, plain, CHOBANI'],
        'source': [usda, usda, usda, usda, usda, usda, np.nan, usda],
    })
    expected = ['packaged', 'generic', 'generic', 'generic', 'unclear',
                'unclear', 'unclear', 'packaged']
    assert_equals(expected, dp.label_tempolife_group(df).tolist())


def test_label_form() -> None:
    """
    Tests label_form.
    """
    names = pd.Series(['Pasta, dry', 'Pasta, cooked',
                       'Soup, condensed, prepared with equal volume water',
                       'Oatmeal, made with water', 'Precooked rice'])
    # 'Precooked' is one word, so it does not count as 'cooked'
    expected = ['as sold', 'prepared', 'prepared', 'prepared', 'as sold']
    assert_equals(expected, dp.label_form(names).tolist())


def test_assign_off_category() -> None:
    """
    Tests assign_off_category, including the priority order for products
    with tags from several categories.
    """
    tags = pd.Series(['en:breakfast-cereals', 'en:meals,en:soups',
                      'en:dairies,en:desserts', 'en:dairies',
                      'en:beverages', np.nan])
    received = dp.assign_off_category(tags)
    assert_equals(['Breakfast cereals', 'Soups', 'Desserts & sweets',
                   'Dairy'], received.iloc[:4].tolist())
    assert_equals(2, int(received.isna().sum()))


def test_load_off() -> None:
    """
    Tests load_off on the filtered sample (written by test_filter_off).
    """
    off = dp.load_off(OFF_SAMPLE_FILTERED)
    assert_equals((6, 10), off.shape)
    assert_equals(['packaged', 'unclear', 'packaged', 'packaged',
                   'packaged', 'packaged'], off['group'].tolist())
    assert_equals(['Breakfast cereals', 'Breads & bakery', 'Soups',
                   'Desserts & sweets', 'Dairy', 'Grains & pasta'],
                  off['food_category'].tolist())
    # 'abc' sugar (pudding) and 150 g salt (cheese) are missing
    assert_equals(1, int(off['sugar'].isna().sum()))
    assert_equals(1, int(off['salt'].isna().sum()))
    assert_equals(30.0, off.loc[0, 'sugar'])


def test_load_tempolife() -> None:
    """
    Tests load_tempolife on the 9-row sample.
    """
    tempolife = dp.load_tempolife(TEMPOLIFE_SAMPLE)
    assert_equals((9, 10), tempolife.shape)
    assert_equals(['packaged', 'generic', 'generic', 'generic', 'unclear',
                   'unclear', 'generic', 'generic', 'generic'],
                  tempolife['group'].tolist())
    assert_equals(['as sold', 'as sold', 'prepared', 'prepared', 'as sold',
                   'as sold', 'as sold', 'as sold', 'as sold'],
                  tempolife['form'].tolist())
    assert_equals('Breakfast cereals', tempolife.loc[0, 'food_category'])
    # Beef and the Estonian berry jam are outside the project categories
    assert_equals(2, int(tempolife['food_category'].isna().sum()))
    assert_equals(60.0, tempolife.loc[8, 'salt'])
    # the berry jam's 0 g sugar is a placeholder (no source), so missing
    assert_equals(True, bool(np.isnan(tempolife.loc[5, 'sugar'])))
    # beef and broth cubes have no sugar in the file, so 3 missing in total
    assert_equals(3, int(tempolife['sugar'].isna().sum()))


def test_drop_exact_duplicates() -> None:
    """
    Tests drop_exact_duplicates: exact copies are removed, but foods that
    share a name and differ in salt are kept.
    """
    df = pd.DataFrame({'dataset': ['T', 'T', 'T', 'O'],
                       'name': ['Artichokes, boiled'] * 2 + ['Pasta, dry'] * 2,
                       'brand': [np.nan] * 4,
                       'sugar': [1.0, 1.0, 2.7, 2.7],
                       'salt': [0.15, 0.74, 0.015, 0.015]})
    assert_equals(4, len(dp.drop_exact_duplicates(df)))
    doubled = pd.concat([df, df])
    assert_equals(df.reset_index(drop=True),
                  dp.drop_exact_duplicates(doubled))


def test_analysis_rows() -> None:
    """
    Tests analysis_rows on the TempoLife sample.
    """
    rows = dp.analysis_rows(dp.load_tempolife(TEMPOLIFE_SAMPLE))
    # Golden Crisp, dry oats, one copy of dry pasta, broth cubes
    assert_equals(['Cereals ready-to-eat, POST, GOLDEN CRISP',
                   'Cereals, oats, regular, dry', 'Pasta, dry',
                   'Soup, chicken broth cubes, dry'], rows['name'].tolist())


def make_foods() -> pd.DataFrame:
    """
    Returns a small DataFrame of foods used by the eda tests.
    """
    return pd.DataFrame({
        'name': ['a', 'a', 'b', 'c', 'd', 'e', 'f', 'g'],
        'dataset': ['Open Food Facts'] * 4 + ['TempoLife'] * 4,
        'food_category': ['Soups'] * 6 + ['Dairy', np.nan],
        'group': ['packaged'] * 4 + ['generic'] * 3 + ['unclear'],
        'sugar': [0.0, 0.0, 2.0, 4.0, 1.0, 3.0, np.nan, 5.0],
        'salt': [1.0, np.nan, 1.0, 1.0, 2.0, 2.0, 2.0, np.nan],
    })


def test_dataset_size() -> None:
    """
    Tests dataset_size.
    """
    assert_equals((8, 6), eda.dataset_size(make_foods()))
    assert_equals((0, 0), eda.dataset_size(pd.DataFrame()))


def test_missing_summary() -> None:
    """
    Tests missing_summary.
    """
    summary = eda.missing_summary(make_foods(), ['sugar', 'salt', 'name'])
    assert_equals([1, 2, 0], summary['missing'].tolist())
    assert_equals([12.5, 25.0, 0.0], summary['percent_missing'].tolist())


def test_missing_by_category() -> None:
    """
    Tests missing_by_category (the food with no category is left out).
    """
    percent = eda.missing_by_category(make_foods(), ['sugar', 'salt'])
    assert_equals([100.0, 0.0], percent.loc['Dairy'].tolist())
    assert_equals(round(100 / 6, 1), percent.loc['Soups', 'salt'])


def test_seven_number_summary() -> None:
    """
    Tests seven_number_summary overall and by group.
    """
    df = pd.DataFrame({'g': ['x', 'x', 'x', 'x', 'y'],
                       'v': [1.0, 2.0, 3.0, 4.0, 10.0]})
    overall = eda.seven_number_summary(df.iloc[:4], 'v')
    # sample standard deviation of 1, 2, 3, 4 is sqrt(5 / 3)
    assert_equals([4.0, 2.5, round((5 / 3) ** 0.5, 3), 1.0, 1.75, 2.5,
                   3.25, 4.0], overall.iloc[0].tolist())
    grouped = eda.seven_number_summary(df, 'v', ['g'])
    assert_equals(10.0, grouped.loc['y', 'max'])
    assert_equals(1.0, grouped.loc['y', 'count'])


def test_category_counts() -> None:
    """
    Tests category_counts, which keeps missing values as their own value.
    """
    counts = eda.category_counts(make_foods(), 'food_category')
    assert_equals(6, counts['Soups'])
    assert_equals(1, counts[np.nan])


def test_group_sizes_and_qualifying() -> None:
    """
    Tests group_sizes and qualifying_categories.
    """
    foods = make_foods()
    sizes = eda.group_sizes(foods, 'sugar')
    assert_equals(4, sizes.loc['Soups', 'packaged'])
    assert_equals(2, sizes.loc['Soups', 'generic'])
    assert_equals(['Soups'], eda.qualifying_categories(foods, 'sugar', 2))
    assert_equals([], eda.qualifying_categories(foods, 'sugar', 3))
    assert_equals([], eda.qualifying_categories(foods[foods['group'] ==
                                                      'generic'], 'sugar'))


def test_assumption_checks() -> None:
    """
    Tests assumption_checks on the Soups rows.
    """
    checks = eda.assumption_checks(make_foods(), 'sugar')
    packaged = checks.loc[('Soups', 'packaged')]
    # packaged sugar: 0, 0, 2, 4 -> median 1, quartiles 0 and 2.5
    assert_equals([4.0, 0.5, 1.0, 2.5, 0.5], packaged.tolist())
    generic = checks.loc[('Soups', 'generic')]
    # generic sugar: 1, 3 -> median 2, quartiles 1.5 and 2.5
    assert_equals([2.0, 0.0, 2.0, 1.0, 0.0], generic.tolist())


def test_comparison_label() -> None:
    """
    Tests comparison_label.
    """
    labels = eda.comparison_label(make_foods())
    assert_equals('Packaged (Open Food Facts)', labels.iloc[0])
    assert_equals('Generic (TempoLife)', labels.iloc[4])


def main():
    test_off_tag_pattern()
    test_filter_off()
    test_clean_nutrients()
    test_count_impossible()
    test_label_tempolife_group()
    test_label_form()
    test_assign_off_category()
    test_load_off()
    test_load_tempolife()
    test_drop_exact_duplicates()
    test_analysis_rows()
    test_dataset_size()
    test_missing_summary()
    test_missing_by_category()
    test_seven_number_summary()
    test_category_counts()
    test_group_sizes_and_qualifying()
    test_assumption_checks()
    test_comparison_label()
    os.remove(OFF_SAMPLE_FILTERED)
    print('All tests passed!')


if __name__ == '__main__':
    main()
