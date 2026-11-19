"""
Linh-An Tran
CSE 163
Exploratory data analysis for the sugar and salt project. For each dataset
(TempoLife, and Open Food Facts once it has been filtered with
data_processing.py) this program reports the size, missing data, summaries
of the variables of interest, and checks of the permutation test's
assumptions. It saves tables to results/ and interactive Plotly charts to
figures/ (HTML, plus PNG copies for the report).
"""
import os

import pandas as pd
import plotly.express as px

import data_processing as dp

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
MIN_GROUP_SIZE = 30
GROUP_COLORS = {'packaged': '#D55E00', 'generic': '#0072B2',
                'unclear': '#BBBBBB'}
SOURCE_COLORS = {'Packaged (Open Food Facts)': '#D55E00',
                 'Packaged (TempoLife)': '#E69F00',
                 'Generic (TempoLife)': '#0072B2'}
CATEGORY_ORDER = [name for name, _, _ in dp.CATEGORY_MAP]


def dataset_size(df: pd.DataFrame) -> tuple[int, int]:
    """
    Takes a DataFrame and returns its (number of rows, number of columns).
    """
    return df.shape


def missing_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Takes a DataFrame and a list of column names and returns a DataFrame
    with one row per column giving the number of missing values
    ('missing') and the percentage of rows missing ('percent_missing').
    """
    missing = df[columns].isna().sum()
    return pd.DataFrame({
        'missing': missing,
        'percent_missing': (missing / len(df) * 100).round(1),
    })


def missing_by_category(df: pd.DataFrame,
                        columns: list[str]) -> pd.DataFrame:
    """
    Takes a DataFrame with a 'food_category' column and a list of column
    names and returns the percentage of missing values in each column for
    each food category (one row per category).
    """
    is_missing = df[columns].isna()
    is_missing['food_category'] = df['food_category']
    return (is_missing.groupby('food_category')[columns].mean() * 100).round(1)


def seven_number_summary(df: pd.DataFrame, column: str,
                         by: list[str] | None = None) -> pd.DataFrame:
    """
    Takes a DataFrame, a numeric column name, and an optional list of
    columns to group by. Returns the count of non-missing values and the
    seven-number summary (mean, std, min, 25%, 50%, 75%, max) of the
    column, overall or for each group, rounded to 3 decimals.
    """
    if by is None:
        return df[column].describe().to_frame().T.round(3)
    return df.groupby(by)[column].describe().round(3)


def category_counts(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Takes a DataFrame and a categorical column name and returns how many
    rows take each value of that column, including missing values.
    """
    return df[column].value_counts(dropna=False)


def group_sizes(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """
    Takes a DataFrame of foods with 'food_category' and 'group' columns
    and returns a table (category rows, group columns) counting the foods
    that have a non-missing value in value_column.
    """
    has_value = df[df[value_column].notna()]
    return pd.crosstab(has_value['food_category'], has_value['group'])


def qualifying_categories(df: pd.DataFrame, value_column: str,
                          min_size: int = MIN_GROUP_SIZE) -> list[str]:
    """
    Takes a DataFrame of foods and a value column and returns a sorted list
    of the food categories that have at least min_size packaged foods and
    at least min_size generic foods with a non-missing value. Returns an
    empty list if either group is absent.
    """
    sizes = group_sizes(df, value_column)
    if 'packaged' not in sizes or 'generic' not in sizes:
        return []
    enough = (sizes['packaged'] >= min_size) & (sizes['generic'] >= min_size)
    return sorted(sizes.index[enough])


def assumption_checks(df: pd.DataFrame, value_column: str,
                      group_column: str = 'group') -> pd.DataFrame:
    """
    Takes a DataFrame of foods, a value column, and the column holding the
    groups to compare. Returns, for each food category and group, the
    number of foods with a value ('n'), the share of values that are
    exactly zero ('share_zero', ties that make median confidence intervals
    coarse), the median, the interquartile range ('iqr'), and the share of
    names that appear more than once in the category and group
    ('share_repeated_name', a warning sign for non-independent rows).
    """
    rows = df[df[value_column].notna()]
    grouped = rows.groupby(['food_category', group_column])
    values = grouped[value_column]
    repeated = rows.duplicated(['food_category', group_column, 'name'],
                               keep=False)
    return pd.DataFrame({
        'n': values.size(),
        'share_zero': values.apply(lambda v: (v == 0).mean()).round(3),
        'median': values.median().round(3),
        'iqr': (values.quantile(0.75) - values.quantile(0.25)).round(3),
        'share_repeated_name': repeated.groupby(
            [rows['food_category'], rows[group_column]]).mean().round(3),
    })


def comparison_label(df: pd.DataFrame) -> pd.Series:
    """
    Takes a DataFrame of foods with 'group' and 'dataset' columns and
    returns labels such as 'Packaged (Open Food Facts)' that name both the
    group and the dataset a food came from.
    """
    return df['group'].str.capitalize() + ' (' + df['dataset'] + ')'


def save_figure(fig, name: str) -> None:
    """
    Saves the Plotly figure fig to figures/ as an interactive HTML file
    and as a PNG image, both called name.
    """
    fig.write_html(os.path.join(FIGURES_DIR, name + '.html'))
    fig.write_image(os.path.join(FIGURES_DIR, name + '.png'), scale=2)


def plot_group_counts(df: pd.DataFrame, title: str, name: str) -> None:
    """
    Takes a DataFrame of labelled foods and saves a stacked bar chart of
    how many foods in each project category are packaged, generic, or
    unclear, with a dashed line at the minimum group size.
    """
    in_category = df[df['food_category'].notna()]
    counts = (in_category.groupby(['food_category', 'group'])
              .size().reset_index(name='foods'))
    fig = px.bar(counts, x='food_category', y='foods', color='group',
                 color_discrete_map=GROUP_COLORS, title=title,
                 category_orders={'food_category': CATEGORY_ORDER,
                                  'group': list(GROUP_COLORS)},
                 labels={'food_category': 'Food category',
                         'foods': 'Number of foods', 'group': 'Label'})
    fig.add_hline(y=MIN_GROUP_SIZE, line_dash='dash',
                  annotation_text='30 foods',
                  annotation_position='top left')
    fig.update_layout(template='simple_white', width=900, height=500)
    save_figure(fig, name)


def plot_missingness(df: pd.DataFrame, title: str, name: str) -> None:
    """
    Takes a DataFrame of foods and saves a grouped bar chart of the
    percentage of foods in each project category missing sugar and salt.
    """
    in_category = df[df['food_category'].notna()]
    percent = missing_by_category(in_category, ['sugar', 'salt'])
    long = percent.reset_index().melt(id_vars='food_category',
                                      var_name='nutrient',
                                      value_name='percent_missing')
    fig = px.bar(long, x='food_category', y='percent_missing',
                 color='nutrient', barmode='group', title=title,
                 category_orders={'food_category': CATEGORY_ORDER},
                 color_discrete_map={'sugar': '#CC79A7', 'salt': '#009E73'},
                 labels={'food_category': 'Food category',
                         'percent_missing': '% of foods missing',
                         'nutrient': 'Nutrient'})
    fig.update_layout(template='simple_white', width=900, height=450)
    save_figure(fig, name)


def plot_box(df: pd.DataFrame, value_column: str, color_column: str,
             colors: dict[str, str], title: str, name: str) -> None:
    """
    Takes a DataFrame of foods and saves box plots of value_column (grams
    per 100 g) for each food category, split by color_column. Each
    category gets its own panel and y-axis so small and large values are
    both readable. Hovering over an outlier shows the food's name.
    """
    present = [c for c in CATEGORY_ORDER
               if c in set(df['food_category'].dropna())]
    fig = px.box(df.dropna(subset=[value_column]), x=color_column,
                 y=value_column, color=color_column, facet_col='food_category',
                 facet_col_wrap=5, facet_row_spacing=0.12,
                 color_discrete_map=colors,
                 category_orders={'food_category': present,
                                  color_column: list(colors)},
                 hover_name='name', points='outliers', title=title,
                 labels={value_column: value_column + ' (g/100 g)',
                         color_column: ''})
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_xaxes(showticklabels=False, title='')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
    fig.update_layout(template='simple_white', width=1150,
                      height=280 * ((len(present) + 4) // 5) + 140,
                      legend={'orientation': 'h', 'y': -0.06})
    save_figure(fig, name)


def describe_dataset(df: pd.DataFrame, title: str) -> str:
    """
    Takes a labelled DataFrame for one dataset and returns a text report
    of its size, missing data, categorical counts, and seven-number
    summaries of sugar and salt.
    """
    rows, columns = dataset_size(df)
    in_category = df[df['food_category'].notna()]
    analysis = dp.analysis_rows(df)
    parts = [
        '=' * 70, title, '=' * 70,
        f'Rows: {rows}  Columns: {columns}',
        '\nMissing values (all rows):',
        missing_summary(df, ['sugar', 'salt', 'brand', 'source']).to_string(),
        '\nLabel counts (all rows):', category_counts(df, 'group').to_string(),
        '\nForm counts (all rows):', category_counts(df, 'form').to_string(),
        '\nFood category counts (all rows):',
        category_counts(df, 'food_category').to_string(),
        '\nMissing % by project category:',
        missing_by_category(in_category, ['sugar', 'salt']).to_string(),
        f'\nRows used in comparisons (project category, packaged/generic, '
        f'as sold, no exact duplicates): {len(analysis)}',
        '\nSugar, all rows:',
        seven_number_summary(df, 'sugar').to_string(),
        '\nSalt, all rows:',
        seven_number_summary(df, 'salt').to_string(),
        '\nSugar by category and group (comparison rows):',
        seven_number_summary(analysis, 'sugar',
                             ['food_category', 'group']).to_string(),
        '\nSalt by category and group (comparison rows):',
        seven_number_summary(analysis, 'salt',
                             ['food_category', 'group']).to_string(),
        '\nCategories with >= 30 packaged and >= 30 generic (sugar): ' +
        str(qualifying_categories(analysis, 'sugar')),
        'Categories with >= 30 packaged and >= 30 generic (salt): ' +
        str(qualifying_categories(analysis, 'salt')),
    ]
    return '\n'.join(parts)


def analyze_tempolife(tempolife: pd.DataFrame) -> str:
    """
    Takes the labelled TempoLife DataFrame, saves its figures, and returns
    its text report including assumption checks.
    """
    analysis = dp.analysis_rows(tempolife)
    plot_group_counts(tempolife, 'TempoLife: packaged vs. generic foods per '
                      'category', 'tempolife_group_counts')
    plot_missingness(tempolife, 'TempoLife: share of foods missing sugar or '
                     'salt', 'tempolife_missingness')
    for nutrient in ['sugar', 'salt']:
        plot_box(analysis, nutrient, 'group', GROUP_COLORS,
                 f'TempoLife: {nutrient} per 100 g, packaged vs. generic',
                 f'tempolife_{nutrient}_box')
    report = describe_dataset(tempolife, 'TempoLife Food Nutrition Dataset')
    for nutrient in ['sugar', 'salt']:
        report += (f'\n\nAssumption checks, {nutrient} (TempoLife):\n' +
                   assumption_checks(analysis, nutrient).to_string())
    return report


def analyze_off_and_combined(off: pd.DataFrame,
                             tempolife: pd.DataFrame) -> str:
    """
    Takes the labelled OFF and TempoLife DataFrames, saves the OFF and
    combined figures, and returns the OFF and combined text reports. The
    combined data holds packaged OFF foods plus all TempoLife comparison
    foods, labelled by group and dataset.
    """
    plot_group_counts(off, 'Open Food Facts: products per category',
                      'off_group_counts')
    plot_missingness(off, 'Open Food Facts: share of products missing sugar '
                     'or salt', 'off_missingness')
    combined = pd.concat([dp.analysis_rows(off), dp.analysis_rows(tempolife)],
                         ignore_index=True)
    combined['comparison'] = comparison_label(combined)
    for nutrient in ['sugar', 'salt']:
        plot_box(combined, nutrient, 'comparison', SOURCE_COLORS,
                 f'{nutrient.capitalize()} per 100 g by category and source',
                 f'combined_{nutrient}_box')
    report = describe_dataset(off, 'Open Food Facts (filtered)')
    main_rows = combined[combined['comparison'] != 'Packaged (TempoLife)']
    report += '\n\n' + '=' * 70 + '\nCombined: packaged OFF vs. generic '
    report += 'TempoLife\n' + '=' * 70
    for nutrient in ['sugar', 'salt']:
        report += (f'\n\nGroup sizes, {nutrient}:\n' +
                   pd.crosstab(combined.dropna(subset=[nutrient])
                               ['food_category'],
                               combined['comparison']).to_string())
        report += (f'\n\nAssumption checks, {nutrient} (main comparison):\n'
                   + assumption_checks(main_rows, nutrient).to_string())
        report += (f'\nQualifying categories ({nutrient}): ' +
                   str(qualifying_categories(main_rows, nutrient)))
    return report


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    tempolife = dp.load_tempolife(dp.TEMPOLIFE_PATH)
    report = analyze_tempolife(tempolife)
    if os.path.exists(dp.OFF_FILTERED_PATH):
        off = dp.load_off(dp.OFF_FILTERED_PATH)
        report += '\n\n' + analyze_off_and_combined(off, tempolife)
    else:
        report += ('\n\nOpen Food Facts not analyzed: run data_processing.py '
                   'first to create ' + dp.OFF_FILTERED_PATH)
    with open(os.path.join(RESULTS_DIR, 'eda_summary.txt'), 'w') as f:
        f.write(report)
    print(report)


if __name__ == '__main__':
    main()
