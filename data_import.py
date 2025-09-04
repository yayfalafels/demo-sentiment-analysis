#!/usr/bin/env python3
"""import and clean-up case data
"""
# dependencies ---------------------------------------------
import pandas as pd


# constants --------------------------------------------------
RAW_EXCEL_FILE = 'Complaintcasesall.xls'
CLEAN_DATA_FILE = 'case_data_clean.csv'
FIRST_YEAR = 2020
YEARS = [FIRST_YEAR + i for i in range(5)]


# module variables --------------------------------------------
case_data_df = None


# data import ------------------------------------------------
def data_cleanup():
    # 01 read from excel
    year_dfs = []
    column_ref = ''
    for year in YEARS:
        df = pd.read_excel(RAW_EXCEL_FILE, sheet_name=str(year))
        year_columns = ','.join(list(df.columns))

        if year == FIRST_YEAR:
            column_ref = year_columns
        else:
            assert column_ref == year_columns, f'column mismatch ref for year {year}'

        df['year'] = year
        year_dfs.append(df)
        print(f'found {len(df)} rows for year {year}')
        print(f'columns: {df.columns}')

    # 02 consolidate single table by year
    case_data_df = pd.concat(year_dfs)

    # 03 export to CSV
    case_data_df.to_csv(CLEAN_DATA_FILE, index=False)

    print(f'saved consolidated case data by year to {CLEAN_DATA_FILE}.')

# entry point -------------------------------------------------
def run():
    print('case complaint sentiment analysis:')
    data_cleanup()


if __name__ == '__main__':
    run()