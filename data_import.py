#!/usr/bin/env python3
"""import and clean-up case data
"""
# dependencies ---------------------------------------------
import logging
import pandas as pd
import re


# constants --------------------------------------------------
RAW_EXCEL_FILE = 'Complaintcasesall.xls'
CLEAN_DATA_FILE = 'case_data_clean.csv'
CURATED_DATA_FILE = 'curated_data.csv'
FIRST_YEAR = 2020
YEARS = [FIRST_YEAR + i for i in range(5)]
FREE_TEXT_FIELD = 'TEST Free text'
WARRINER_DATA_URL = 'https://github.com/JULIELab/XANEW/blob/master/Ratings_Warriner_et_al.csv'
WARRINER_DATA_CSV = 'Ratings_Warriner_et_al.csv' # manually download CSV from URL and save to local folder

WARRINER_COLUMNS = {
    'Word': 'word', 
    'V.Mean.Sum': 'valence', 
    'A.Mean.Sum': 'arousal'
}

# module variables --------------------------------------------
case_data_df = None
warriner_data_df = None
valence_dict = {}
arousal_dict = {}


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


def data_load():
    global case_data_df, warriner_data_df
    case_data_df = pd.read_csv(CLEAN_DATA_FILE)
    warriner_data_df = pd.read_csv(WARRINER_DATA_CSV, index_col=0)


def data_export(curated_data_df):
    curated_data_df.to_csv(CURATED_DATA_FILE)


def lexicon_load():
    global valence_dict, arousal_dict 
    lexicon = warriner_data_df[list(WARRINER_COLUMNS.keys())].rename(
        columns=WARRINER_COLUMNS
    )

    # Convert to dictionary for fast lookup
    valence_dict = dict(zip(lexicon.word, lexicon.valence))
    arousal_dict = dict(zip(lexicon.word, lexicon.arousal))


def score_valence_arousal(text):
    words = re.findall(r"\b\w+\b", text.lower())
    
    v_scores, a_scores = [], []
    for w in words:
        if w in valence_dict:
            v_scores.append(valence_dict[w])
            a_scores.append(arousal_dict[w])
    
    if not v_scores:  # No matches
        return {"valence": None, "arousal": None, "word_count": len(words)}
    
    return {
        "valence": sum(v_scores) / len(v_scores),
        "arousal": sum(a_scores) / len(a_scores),
        "word_count": len(words),
        "matched_words": len(v_scores)
    }


def sentiment_analysis():
    data_load()

    logging.info(f'Complaint case data:')
    print(case_data_df.head())

    logging.info(f'Warriner emotions data set:')
    print(warriner_data_df.head())

    logging.info(f'running sentiment analysis')
    lexicon_load()
    df = case_data_df.copy()
    scores = df[FREE_TEXT_FIELD].apply(score_valence_arousal).apply(pd.Series)
    scores = scores.add_prefix('warr_')
    df = df.join(scores)

    data_export(df)
    logging.info(f'processed data:')
    print(df.head())


# entry point -------------------------------------------------
def run():
    print('case complaint sentiment analysis:')
    #data_cleanup()
    sentiment_analysis()


if __name__ == '__main__':
    run()