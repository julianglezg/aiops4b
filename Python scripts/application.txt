# Import libraries

import pandas as pd

from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.framework.templates.autogen.forecast_config import ForecastConfig, MetadataParam, ModelComponentsParam, \
    EvaluationPeriodParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.common import constants as cst

from plotly import graph_objs as go

import warnings
warnings.filterwarnings("ignore")

from ml_utils import *

import nltk
nltk.download('stopwords')
nltk.download('rslp')

from nltk.corpus import stopwords
import re
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import statsmodels.api as sm

# We read the metrics file and the reviews dataset

with open(f'metrics_pvalue_ord.csv', 'rb') as f:
    data = f.read()

metrics = pd.read_csv(f'metrics_pvalue_ord.csv',index_col=[0])
#metrics = metrics.iloc[:609,:]

with open(f'olist_order_reviews_dataset.csv', 'rb') as f:
    data = f.read()

reviews = pd.read_csv(f'olist_order_reviews_dataset.csv',index_col=[0])

# These are the parameters that have been explained in Chapter 3

anomaly_start = "2017-11-23"
anomaly_end = "2017-11-26"
country = "Brazil"
prediction_days = 5



# 1. Forecasting model

# 1. Anomalies
anomaly_df = pd.DataFrame({
    # start and end are included
    cst.START_DATE_COL: [anomaly_start],
    cst.END_DATE_COL: [anomaly_end],
    cst.ADJUSTMENT_DELTA_COL: [np.nan]
})

anomaly_info = {
    "value_col": "Value",
    "anomaly_df": anomaly_df,
    "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL
}

# 2. Growth

growth = {
    "growth_term": "linear"
}

# 3. Changepoint detection

changepoints = {
    "changepoints_dict": dict(
        method="auto",
        yearly_seasonality_order=2,
        regularization_strength=0.6,
        resample_freq="7D",
        potential_changepoint_n=25,
        yearly_seasonality_change_freq="365D",
        no_changepoint_proportion_from_end=0.1
    )
}

# 4. Seasonality

yearly_seasonality_order = 2
weekly_seasonality_order = 4
seasonality = {
    "yearly_seasonality": yearly_seasonality_order,
    "quarterly_seasonality": False,
    "monthly_seasonality": False,
    "weekly_seasonality": weekly_seasonality_order,
    "daily_seasonality": False
}

# 5. Holidays and events

events = {
    "holidays_to_model_separately": SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES,
    "holiday_lookup_countries": [country],  # only look up holidays in Brazil
    "holiday_pre_num_days": 2,  # also mark the 2 days before a holiday as holiday
    "holiday_post_num_days": 2,  # also mark the 2 days after a holiday as holiday
}

# Complete model

metadata = MetadataParam(
    time_col="Time_col",
    value_col="Value",
    freq="D",
    anomaly_info=anomaly_info,
)

model_components = ModelComponentsParam(
    seasonality=seasonality,
    growth=growth,
    events=events,
    changepoints=changepoints,
)

evaluation_period = EvaluationPeriodParam(
    test_horizon=28,
    cv_horizon=prediction_days,
    cv_max_splits=3,
    cv_min_train_periods=300
)

# running the forecast
forecaster = Forecaster()

result = forecaster.run_forecast_config(
    df=metrics,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=prediction_days,
        coverage=0.95,
        metadata_param=metadata,
        model_components_param=model_components,
        evaluation_period_param=evaluation_period
    )
)

# fig2 is the forecasting graph, which has not been included in the app

forecast = result.forecast
fig2 = forecast.plot()
fig2 = go.Figure(fig2)

from greykite.framework.utils.result_summary import summarize_grid_search_results

# this applies cross validation and outputs the test MAPE

cv_results = summarize_grid_search_results(
    grid_search=result.grid_search,
    decimals=1,
    cv_report_metrics=None,
    column_order=["rank", "mean_test", "split_test", "params"])
cv_results["params"] = cv_results["params"].astype(str)
cv_results.set_index("params", drop=True, inplace=True)
print(cv_results.transpose())

# forecast_df contains the revenue forecast dataframe, from which we will display 5 elements

forecast_df = forecast.df

df = pd.DataFrame(forecast_df.iloc[((-prediction_days) - 1):-1, 2]).reset_index(drop=True)
df.index += 1


# 2.1 Recommendation - analyse. This is the changepoint detection model

# this is the dataset metadata
metadata = dict(
    time_col="Time_col",
    value_col="Value",
    freq="D"  # our frequency is daily
)
# these are the changepoint parameters
model_components = dict(
    changepoints={
        "changepoints_dict": {
            "method": "auto",
            "yearly_seasonality_order": 15,
            "regularization_strength": 0.6,
            "resample_freq": "7D",
            "potential_changepoint_n": 25,
            "no_changepoint_proportion_from_end": 0.05
        }
    },
    custom={
        "fit_algorithm_dict": {
            "fit_algorithm": "ridge"}})  # we use ridge to prevent overfitting

config = ForecastConfig.from_dict(
    dict(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=5,
        coverage=0.95,
        metadata_param=metadata,
        model_components_param=model_components))

# the forecaster is run with the changepoints configuration
forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=metrics,
    config=config)

import plotly

plotly.io.renderers.default = 'browser'

# this fig is the changepoint graph that we can see in AIOps4B

fig = result.model[-1].plot_trend_changepoint_detection(dict(plot=False))
#print(fig.show())

fig = go.Figure(fig)


# 2.2 Root-Cause Analysis and Deep Analysis

#reviews = pd.read_csv(reviews_path)

## START - Adapted from https://www.kaggle.com/thiagopanini/e-commerce-sentiment-analysis-eda-viz-nlp ##

## 1. REVIEW COMMENTS DATASET

reviews = reviews.sort_values(by='review_answer_timestamp')
df_comments = reviews.loc[:, ['review_score', 'review_comment_message']]
df_comments = df_comments.iloc[99332:, :]
df_comments = df_comments.dropna(subset=['review_comment_message'])
df_comments = df_comments.reset_index(drop=True)
df_comments.columns = ['score', 'comment']

# print(df_comments.head())

## 2. REGULAR EXPRESSIONS

# we remove line breaks

def re_breakline(text_list):
    return [re.sub('[\n\r]', ' ', r) for r in text_list]

reviews = list(df_comments['comment'].values)

reviews_breakline = re_breakline(reviews)
df_comments['re_breakline'] = reviews_breakline

## 3. HYPERLINKS

# we replace hyperlinks with "link"

def re_hyperlinks(text_list):
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, ' link ', r) for r in text_list]

reviews_hyperlinks = re_hyperlinks(reviews_breakline)
df_comments['re_hyperlinks'] = reviews_hyperlinks

## 4. DATES

# we replace dates with "data"

def re_dates(text_list):
    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(pattern, ' data ', r) for r in text_list]

reviews_dates = re_dates(reviews_hyperlinks)
df_comments['re_dates'] = reviews_dates

## 5. MONEY

# we replace currency with "dinheiro", the Portuguese word for money

def re_money(text_list):
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(pattern, ' dinheiro ', r) for r in text_list]

reviews_money = re_money(reviews_dates)
df_comments['re_money'] = reviews_money

## 6. NUMBERS

# we replace numbers with "numero", the Portuguese word for number

def re_numbers(text_list):
    return [re.sub('[0-9]+', ' numero ', r) for r in text_list]

reviews_numbers = re_numbers(reviews_money)
df_comments['re_numbers'] = reviews_numbers

## 7. NEGATION

# we replace negation or negative expressions with "negação", the Portuguese word for negation

def re_negation(text_list):
    return [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', r) for r in text_list]

reviews_negation = re_negation(reviews_numbers)
df_comments['re_negation'] = reviews_negation

## 8. SPECIAL CHARACTERS

# we remove special characters

def re_special_chars(text_list):
    return [re.sub('\W', ' ', r) for r in text_list]

reviews_special_chars = re_special_chars(reviews_negation)
df_comments['re_special_chars'] = reviews_special_chars

## 9. WHITESPACES

# we remove whitespaces

def re_whitespaces(text_list):
    white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end

reviews_whitespaces = re_whitespaces(reviews_special_chars)
df_comments['re_whitespaces'] = reviews_whitespaces

## 10. STOPWORDS

# we remove Portuguese stopwords

pt_stopwords = stopwords.words('portuguese')

def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
    return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]

reviews_stopwords = [' '.join(stopwords_removal(review)) for review in reviews_whitespaces]
df_comments['stopwords_removed'] = reviews_stopwords

## 11. STEMMING

def stemming_process(text, stemmer=RSLPStemmer()):
    return [stemmer.stem(c) for c in text.split()]

reviews_stemmer = [' '.join(stemming_process(review)) for review in reviews_stopwords]
df_comments['stemming'] = reviews_stemmer

# Feature Extraction

# we define a function for feature extraction

def extract_features_from_corpus(corpus, vectorizer, df=False):
    corpus_features = vectorizer.fit_transform(corpus).toarray()
    features_names = vectorizer.get_feature_names()
    df_corpus_features = None
    if df:
        df_corpus_features = pd.DataFrame(corpus_features, columns=features_names)
    return corpus_features, df_corpus_features

## 1. CountVectorizer

count_vectorizer = CountVectorizer(max_features=300, min_df=7, max_df=0.8, stop_words=pt_stopwords)
countv_features, df_countv_features = extract_features_from_corpus(reviews_stemmer, count_vectorizer, df=True)

## 2. TF-IDF

tfidf_vectorizer = TfidfVectorizer(max_features=300, min_df=7, max_df=0.8, stop_words=pt_stopwords)
tfidf_features, df_tfidf_features = extract_features_from_corpus(reviews_stemmer, tfidf_vectorizer, df=True)

# scores of 1-2 will be considered negative and scores of 3-5 will be considered positive

score_map = {
    1: 'negative',
    2: 'negative',
    3: 'positive',
    4: 'positive',
    5: 'positive'
}
df_comments['sentiment_label'] = df_comments['score'].map(score_map)

def ngrams_count(corpus, ngram_range, n=-1, cached_stopwords=stopwords.words('portuguese')):
    vectorizer = CountVectorizer(stop_words=cached_stopwords, ngram_range=ngram_range).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_list = words_freq[:n]
    count_df = pd.DataFrame(total_list, columns=['ngram', 'count'])
    return count_df

positive_comments = df_comments.query('sentiment_label == "positive"')['stemming']
negative_comments = df_comments.query('sentiment_label == "negative"')['stemming']

# extracting the top 10 unigrams by sentiment
unigrams_pos = ngrams_count(positive_comments, (1, 1), 10)
unigrams_neg = ngrams_count(negative_comments, (1, 1), 10)

# extracting the top 10 bigrams by sentiment
bigrams_pos = ngrams_count(positive_comments, (2, 2), 10)
bigrams_neg = ngrams_count(negative_comments, (2, 2), 10)

# extracting the top 10 trigrams by sentiment
trigrams_pos = ngrams_count(positive_comments, (3, 3), 10)
trigrams_neg = ngrams_count(negative_comments, (3, 3), 10)

## END - adapted from https://www.kaggle.com/thiagopanini/e-commerce-sentiment-analysis-eda-viz-nlp ##

# here we select the dates from the previous changepoint to the next changepoint. Like we explained in Chapter 3,
# the metrics' change will be measured in these periods of time

# from 2017-01-23 to Changepoint 2

d21 = '2017-01-23'
d22 = '2017-07-23'

# from Changepoint 2 to Changepoint 3

d31 = '2017-07-23'
d32 = '2017-11-26'

# from Changepoint 3 to Changepoint 4

d41 = '2017-11-26'
d42 = '2018-07-22'

# these parameters are the calculations for the root-cause analysis

# these are the metrics for 2017-01-23

review1 = 3.69230769230769
freight1 = 18.0077777777777
delivery1 = 0.888888888888888
revenue1 = 7553.89
orders1 = 39

# these are the metrics for the period between 2017-01-23 and Changepoint 2

review2 = pd.DataFrame(metrics.iloc[32:213,4])
freight2 = pd.DataFrame(metrics.iloc[32:213,5])
delivery2 = pd.DataFrame(metrics.iloc[32:213,6])

revenue2 = pd.DataFrame(metrics.iloc[32:213,1])
orders2 = pd.DataFrame(metrics.iloc[32:213,7])

# these are the metrics for the period between Changepoint 2 and Changepoint 3

review3 = pd.DataFrame(metrics.iloc[213:339,4])
freight3 = pd.DataFrame(metrics.iloc[213:339,5])
delivery3 = pd.DataFrame(metrics.iloc[213:339,6])

revenue3 = pd.DataFrame(metrics.iloc[213:339,1])
orders3 = pd.DataFrame(metrics.iloc[213:339,7])

# these are the metrics for the period between Changepoint 3 and Changepoint 4

review4 = pd.DataFrame(metrics.iloc[339:577,4])
freight4 = pd.DataFrame(metrics.iloc[339:577,5])
delivery4 = pd.DataFrame(metrics.iloc[339:577,6])

revenue4 = pd.DataFrame(metrics.iloc[339:577,1])
orders4 = pd.DataFrame(metrics.iloc[339:577,7])

# these are the metrics for the period between Changepoint 4 to the end

review0 = pd.DataFrame(metrics.iloc[577:,4])
freight0 = pd.DataFrame(metrics.iloc[577:,5])
delivery0 = pd.DataFrame(metrics.iloc[577:,6])

revenue0 = pd.DataFrame(metrics.iloc[577:,1])
orders0 = pd.DataFrame(metrics.iloc[577:,7])

# these parameters are the metrics between certain periods of time that will be used in the linear correlation,
# since only 3 data points is not enough

# the metrics for 01-03-17

review5 = metrics.iloc[69,4]
freight5 = metrics.iloc[69,5]
delivery5 = metrics.iloc[69,6]

revenue5 = metrics.iloc[69,1]

# the metrics from 02-03-17 to 15-04-17

review6 = pd.DataFrame(metrics.iloc[70:115,4])
freight6 = pd.DataFrame(metrics.iloc[70:115,5])
delivery6 = pd.DataFrame(metrics.iloc[70:115,6])

revenue6 = pd.DataFrame(metrics.iloc[70:115,1])

# the metrics for 01-03-18

review7 = metrics.iloc[434,4]
freight7 = metrics.iloc[434,5]
delivery7 = metrics.iloc[434,6]

revenue7 = metrics.iloc[434,1]

# the metrics from 02-03-18 to 15-04-18

review8 = pd.DataFrame(metrics.iloc[435:480,4])
freight8 = pd.DataFrame(metrics.iloc[435:480,5])
delivery8 = pd.DataFrame(metrics.iloc[435:480,6])

revenue8 = pd.DataFrame(metrics.iloc[435:480,1])

#####

# now we calculate the change in metrics for Changepoint 2, for the period before and after it

review21 = round((review3.mean()[0]-review2.mean()[0])/review2.mean()[0]*100,2)
freight21 = round((freight3.mean()[0]-freight2.mean()[0])/freight2.mean()[0]*100,2)
delivery21 = round((delivery3.mean()[0]-delivery2.mean()[0])/delivery2.mean()[0]*100,2)
revenue21 = round((revenue3.mean()[0]-revenue2.mean()[0])/revenue2.mean()[0]*100,2)
orders21 = round((orders3.mean()[0]-orders2.mean()[0])/orders2.mean()[0]*100,2)

# now we calculate the change in metrics for Changepoint 3, for the period before and after it

review32 = round((review4.mean()[0]-review3.mean()[0])/review3.mean()[0]*100,2)
freight32 = round((freight4.mean()[0]-freight3.mean()[0])/freight3.mean()[0]*100,2)
delivery32 = round((delivery4.mean()[0]-delivery3.mean()[0])/delivery3.mean()[0]*100,2)
revenue32 = round((revenue4.mean()[0]-revenue3.mean()[0])/revenue3.mean()[0]*100,2)
orders32 = round((orders4.mean()[0]-orders3.mean()[0])/orders3.mean()[0]*100,2)

# now we calculate the change in metrics for Changepoint 4, for the period before and after it

review43 = round((review0.mean()[0]-review4.mean()[0])/review4.mean()[0]*100,2)
freight43 = round((freight0.mean()[0]-freight4.mean()[0])/freight4.mean()[0]*100,2)
delivery43 = round((delivery0.mean()[0]-delivery4.mean()[0])/delivery4.mean()[0]*100,2)
revenue43 = round((revenue0.mean()[0]-revenue4.mean()[0])/revenue4.mean()[0]*100,2)
orders43 = round((orders0.mean()[0]-orders4.mean()[0])/orders4.mean()[0]*100,2)

# these calculations are for the other periods, for the linear regression

review65 = round((review6.mean()[0]-review5)/review5*100,2)
freight65 = round((freight6.mean()[0]-freight5)/freight5*100,2)
delivery65 = round((delivery6.mean()[0]-delivery5)/delivery5*100,2)
revenue65 = round((revenue6.mean()[0]-revenue5)/revenue5*100,2)

review87 = round((review8.mean()[0]-review7)/review7*100,2)
freight87 = round((freight8.mean()[0]-freight7)/freight7*100,2)
delivery87 = round((delivery8.mean()[0]-delivery7)/delivery7*100,2)
revenue87 = round((revenue8.mean()[0]-revenue7)/revenue7*100,2)

# we can now build a dataframe with the metrics changes

df2 = pd.DataFrame({'review':[review21,review32,review43,review65,review87],
                    'freight':[freight21,freight32,freight43,freight65,freight87],
                    'delivery':[delivery21,delivery32,delivery43,delivery65,delivery87],
                    'revenue':[revenue21,revenue32,revenue43,revenue65,revenue87]})

X = df2[['review','freight','delivery']]
Y = df2['revenue']

# we apply a linear regression

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

# these parameters are the linear correlation coefficients that will be present in the dashboard

params = model.params


# 3. Visualization

import dash_table

# Inspired by https://medium.com/analytics-vidhya/building-a-dashboard-app-using-plotlys-dash-a-complete-guide-from-beginner-to-pro-61e890bdc423

def create_data_table(dataframe):
    data_table = dash_table.DataTable(
        id = 'table',
        data = dataframe.to_dict(),
        columns = [{'id':c,'name':c} for c in dataframe.columns],
        style_table = {'overflowY':'scroll'},
        style_cell = {'width':'100px'}
    )


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Tbody([
            html.Tr([
                html.Td(round(dataframe.iloc[i][col])) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
application = app.server

app.layout = html.Div([
    html.Div(
        className="header",
        children=[
            html.Div(
                className="div-info",
                children=[
                    html.H1(className="title", children="AIOps4B - Automated Forecasting, Root-Cause Analysis, and Deep Analysis"),
                ],
            ),
        ],
    ),

    html.H2(["Revenue KPI"]),
    html.P(
        """
        Graphical representation of the historical sales, with their trend and the detected trend changepoints.
        """
    ),

    html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="graph", figure=fig)),
                    dbc.Col([html.H2(str(prediction_days) + '-Day Forecast'),
                             html.Div(generate_table(df))],width=2)
                ]
            ),
        ]
    ),

    dbc.Row(
        [
            dbc.Col([
                html.Label('Select Changepoint:'),
                dcc.Dropdown(
                    id="dropdown-component",
                    options=[
                        {'label': '1.- 27 November 2016', 'value': 'cp1'},
                        {'label': '2.- 23 July 2017', 'value': 'cp2'},
                        {'label': '3.- 26 November 2017', 'value': 'cp3'},
                        {'label': '4.- 22 July 2018', 'value': 'cp4'}
                    ],
                    value='cp4'
                ),

                html.P(
                    className="root-cause", id="root-cause", children=[""]
                )
            ]),
            dbc.Col(
                html.P(
                    className="recommendations", id="recommendations", children=[""]
                )
            )
        ]
    ),
])

# Callbacks #

# Inspired by https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-daq-satellite-dashboard/app.py

@app.callback(
    Output("root-cause", "children"),
    [Input("dropdown-component", "value")],
)
def update_root_cause(val):
    if val == "cp1":
        text = (
            "-"
        )
    elif val == "cp2":
        text = (html.P(["The trend is UP.",html.Br(),
                        "The change in average REVENUE for Changepoint 2 is of {}".format(revenue21), " %.",
                        html.Br(),
                        "The change in ORDERS for Changepoint 2 is of {}".format(orders21), " %.",
                        ]),
                html.H2("Root-Cause Analysis"),
                html.H3("Impact of Metrics on the Trend:"),
                html.P(["Average Review Score: {}%".format(round(params[0]*review21, 2)),html.Br(),
                        "Average Freight Value: {}%".format(round(params[1]*freight21, 2)),html.Br(),
                        "Average Delivery Score: {}%".format(round(params[2]*delivery21, 2)),html.Br(),
                        html.Br(),
                        "There is a {}% increase in revenue due to the metrics.".format(round(round(params[0]*review21, 2) + round(params[1]*freight21, 2) + round(params[2]*delivery21, 2), 2))
                        ]),
                html.H3("Impact of Events on the Trend:"),
                html.P(["There is a {}% increase in revenue due to events.".format(round(revenue21-(round(params[0]*review21, 2) + round(params[1]*freight21, 2) + round(params[2]*delivery21,2)))), html.Br(),
                        "Event: Sales Increase."])
                )

    elif val == "cp3":
        text = (html.P(["The trend is UP.",html.Br(),
                        "The change in average EVENUE for Changepoint 3 is of {}".format(revenue32), " %.",
                        html.Br(),
                        "The change in ORDERS for Changepoint 3 is of {}".format(orders32), " %.",
                        ]),
                html.H2("Root-Cause Analysis"),
                html.H3("Impact of Metrics on the Trend:"),
                html.P(["Average Review Score: {}%".format(round(params[0]*review32, 2)),html.Br(),
                        "Average Freight Value: {}%".format(round(params[1]*freight32, 2)),html.Br(),
                        "Average Delivery Score: {}%".format(round(params[2]*delivery32, 2)),html.Br(),
                        html.Br(),
                        "There is a {}% increase in revenue due to the metrics.".format(round(round(params[0]*review32, 2) + round(params[1]*freight32, 2) + round(params[2]*delivery32, 2), 2))
                        ]),
                html.H3("Impact of Events on the Trend:"),
                html.P(
                    ["There is a {}% increase in revenue due to events.".format(round(revenue32-(round(params[0]*review32, 2) + round(params[1]*freight32, 2) + round(params[2]*delivery32,2)))), html.Br(),
                     "Event: Black Friday."])
                )

    elif val == "cp4":
        text = (html.P(["The trend is DOWN.",html.Br(),
                        "The change in average REVENUE for Changepoint 4 is of {}".format(revenue43), " %.",
                        html.Br(),
                        "The change in ORDERS for Changepoint 4 is of {}".format(orders43), " %.",
                        ]),
                html.H2("Root-Cause Analysis"),
                html.H3("Impact of Metrics on the Trend:"),
                html.P(["Average Review Score: {}%".format(round(params[0]*review43, 2)),html.Br(),
                        "Average Freight Value: {}%".format(round(params[1]*freight43, 2)),html.Br(),
                        "Average Delivery Score: {}%".format(round(params[2]*delivery43, 2)),html.Br(),
                        html.Br(),
                        "There is a {}% decrease in revenue due to the metrics.".format(round(round(params[0]*review43, 2) + round(params[1]*freight43, 2) + round(params[2]*delivery43, 2), 2))
                        ]),
                html.H3("Impact of Events on the Trend:"),
                html.P(
                    ["There is a {}% increase in revenue due to unknown events.".format(round(revenue43-(round(params[0]*review43, 2) + round(params[1]*freight43, 2) + round(params[2]*delivery43,2)))), html.Br(),
                     "Event: Sales Decrease."])
                )
    return text

@app.callback(
    Output("recommendations", "children"),
    [Input("dropdown-component", "value")],
)
def update_recommendations(val):

    if val == "cp1":
        text = ("")

    elif val == "cp2":
        text = (html.P(["The correlation coefficient of the average review score is {}.".format(round(params[0],2)),
                        html.Br(),
                        "The correlation coefficient of the average freight value is {}.".format(round(params[1],2)),
                        html.Br(),
                        "The correlation coefficient of the average delivery score is {}.".format(round(params[2],2))]))


    elif val == "cp3":
        text = (html.P(["The correlation coefficient of the average review score is {}.".format(round(params[0],2)),
                        html.Br(),
                        "The correlation coefficient of the average freight value is {}.".format(round(params[1],2)),
                        html.Br(),
                        "The correlation coefficient of the average delivery score is {}.".format(round(params[2],2))]))


    elif val == "cp4":
        text = (html.P(["The correlation coefficient of the average review score is {}.".format(round(params[0],2)),
                        html.Br(),
                        "The correlation coefficient of the average freight value is {}.".format(round(params[1],2)),
                        html.Br(),
                        "The correlation coefficient of the average delivery score is {}.".format(round(params[2],2))]),
                html.H2(["Deep Analysis"]),
                html.H3(['Product Reviews (most common problems):']),

                html.P(["{} customers did not receive their product.".format(trigrams_neg.loc[trigrams_neg['ngram'] == 'neg receb produt', 'count'].iloc[0]),
                        html.Br(),
                        "{} customers had a problem with the tracking code.".format(trigrams_neg.loc[trigrams_neg['ngram'] == 'produt códig vem', 'count'].iloc[0]),
                        html.Br(),
                        "{} customers received a different product than the one they ordered.".format(trigrams_neg.loc[trigrams_neg['ngram'] == 'outr total difer', 'count'].iloc[0])]),

                html.H3(["Delivery Score (most common problems):"]),
                html.P(["Orders to the North and Northeastern regions have a longer delivery time."]),

                html.H3(["Delivery Cost (most common problems):"]),
                html.P(["Orders to the North and Northeastern regions have a higher delivery cost."])
                )

    return text

# now that we have defined the Dash app, we run it in port 80

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port='80')


